"""Experimental trajectory-level fused hedger with explicit first-order reverse pass.

Each program owns one path and accumulates its parameter gradients. A final
JAX reduction sums independent path contributions. This is deliberately an
isolated benchmark candidate, not a replacement for higher-order teacher AD.
"""
from functools import partial

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def forward(parameters, prices, cost, cap, premium, strike, historical):
    """Return losses [B] and saved GRU/accounting tape [B,T,6H+8] from log prices."""
    batch, n_dates = prices.shape
    n_steps = n_dates-1
    hidden = parameters[1].shape[1]
    assert hidden in (32, 64)
    rows_size = 4*hidden
    tape_size = 6*hidden+8

    def kernel(wir, whr, br, bnr, wor, bor, lr, lossr, tr):
        """Advance one path entirely on device, storing the exact gates for reversal."""
        path = pl.program_id(0)
        rows, cols, features_index = jnp.arange(rows_size), jnp.arange(hidden), jnp.arange(4)
        wi = pl.load(wir, (rows[:, None], features_index[None, :]), mask=rows[:, None]<3*hidden, other=0.)
        wh = pl.load(whr, (rows[:, None], cols[None, :]), mask=rows[:, None]<3*hidden, other=0.)
        bias = pl.load(br, (rows,), mask=rows<3*hidden, other=0.)
        bias_n = pl.load(bnr, (cols,))
        wo = pl.load(wor, (0, cols))
        bo = pl.load(bor, (0,))
        chex.assert_shape(wi, (rows_size, 4))
        chex.assert_shape(wh, (rows_size, hidden))
        chex.assert_shape((bias_n, wo), (hidden,))

        def step(day, carry):
            """Map day and hidden/shares/cumulative-cost carry to the next day."""
            old_hidden, old_shares, accumulated = carry
            lp = pl.load(lr, (path, day))
            previous = pl.load(lr, (path, jnp.maximum(day-1, 0)))
            price = jnp.exp(lp)
            if historical:
                x = jnp.where(features_index == 0, lp, jnp.where(features_index == 1, previous,
                    jnp.where(features_index == 2, (n_steps-day)/n_steps, old_shares)))
            else:
                x = jnp.where(features_index == 0, lp, jnp.where(features_index == 1, lp-previous,
                    jnp.where(features_index == 2, day/n_steps, old_shares*price/cap)))
            chex.assert_shape(x, (4,))
            ig = jnp.sum(wi*x[None, :], axis=1)+bias
            hg = jnp.sum(wh*old_hidden[None, :], axis=1)
            chex.assert_shape((ig, hg), (rows_size,))
            ir, iz, inn, _ = jnp.split(ig, 4)
            hr, hz, hn, _ = jnp.split(hg, 4)
            reset = jax.nn.sigmoid(ir+hr)
            update = jax.nn.sigmoid(iz+hz)
            new = jnp.tanh(inn+reset*(hn+bias_n))
            following = new+update*(old_hidden-new)
            chex.assert_shape((reset, update, new, following), (hidden,))
            allocation = jax.nn.sigmoid(jnp.sum(wo*following)+bo)
            next_shares = cap*allocation if historical else cap*allocation/price
            traded = (next_shares-old_shares)*price
            accumulated = accumulated+traded+cost*jnp.abs(traded)
            for index, value in enumerate((old_hidden, following, reset, update, new, hn)):
                pl.store(tr, (path, day, index*hidden+cols), value)
            pl.store(tr, (path, day, 6*hidden+features_index), x)
            pl.store(tr, (path, day, 6*hidden+4), old_shares)
            pl.store(tr, (path, day, 6*hidden+5), next_shares)
            pl.store(tr, (path, day, 6*hidden+6), allocation)
            pl.store(tr, (path, day, 6*hidden+7), 0.)
            return following, next_shares, accumulated

        _, shares, accumulated = jax.lax.fori_loop(0, n_steps, step,
            (jnp.zeros(hidden, prices.dtype), jnp.asarray(0., prices.dtype), jnp.asarray(-premium, prices.dtype)))
        terminal_price = jnp.exp(pl.load(lr, (path, n_steps)))
        liquidation = shares*terminal_price
        loss = jnp.maximum(terminal_price-strike, 0.)+accumulated-liquidation+cost*jnp.abs(liquidation)
        pl.store(lossr, (path,), loss)

    outputs = (jax.ShapeDtypeStruct((batch,), prices.dtype),
               jax.ShapeDtypeStruct((batch, n_steps, tape_size), prices.dtype))
    return pl.pallas_call(kernel, out_shape=outputs, grid=(batch,), name='persistent_hedger_forward')(*parameters, prices)


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6))
def losses(parameters, prices, cost, cap, premium, strike, historical):
    """Map six parameter arrays/log prices to losses; static arguments set contract conventions."""
    return forward(parameters, prices, cost, cap, premium, strike, historical)[0]


def forward_rule(parameters, prices, cost, cap, premium, strike, historical):
    """Return losses and save parameter/input/tape PyTree for custom reversal."""
    value, tape = forward(parameters, prices, cost, cap, premium, strike, historical)
    return value, (parameters, prices, tape)


def backward_rule(cost, cap, premium, strike, historical, residual, cotangent):
    """Reverse the financial ledger and GRU through time; return parameter and price gradients."""
    parameters, prices, tape = residual
    batch, n_dates = prices.shape
    n_steps, hidden = n_dates-1, parameters[1].shape[1]
    rows_size = 4*hidden

    def kernel(wir, whr, br, bnr, wor, bor, lr, tr, cr, gwir, gwhr, gbr, gbnr, gwor, gbor, glr):
        """Accumulate one trajectory's reverse recurrence and write disjoint gradient buffers."""
        path = pl.program_id(0)
        rows, cols, features_index = jnp.arange(rows_size), jnp.arange(hidden), jnp.arange(4)
        wi = pl.load(wir, (rows[:, None], features_index[None, :]), mask=rows[:, None]<3*hidden, other=0.)
        wh = pl.load(whr, (rows[:, None], cols[None, :]), mask=rows[:, None]<3*hidden, other=0.)
        bn = pl.load(bnr, (cols,))
        wo = pl.load(wor, (0, cols))
        cot = pl.load(cr, (path,))
        terminal_price = jnp.exp(pl.load(lr, (path, n_steps)))
        terminal_shares = pl.load(tr, (path, n_steps-1, 6*hidden+5))
        share_gradient = cot*terminal_price*(-1+cost)
        terminal_gradient = cot*(jnp.where(terminal_price>strike, terminal_price, 0.)+terminal_shares*terminal_price*(-1+cost))
        pl.store(glr, (path, n_steps), terminal_gradient)

        def step(reverse_day, carry):
            """Reverse one time step, accumulating parameter and previous-state cotangents."""
            gh, gshare, pending_lp, gwi, gwh, gb, gbn, gwo, gbo = carry
            day = n_steps-1-reverse_day
            old_hidden = pl.load(tr, (path, day, cols))
            following = pl.load(tr, (path, day, hidden+cols))
            reset = pl.load(tr, (path, day, 2*hidden+cols))
            update = pl.load(tr, (path, day, 3*hidden+cols))
            new = pl.load(tr, (path, day, 4*hidden+cols))
            hn = pl.load(tr, (path, day, 5*hidden+cols))
            x = pl.load(tr, (path, day, 6*hidden+features_index))
            old_shares = pl.load(tr, (path, day, 6*hidden+4))
            next_shares = pl.load(tr, (path, day, 6*hidden+5))
            allocation = pl.load(tr, (path, day, 6*hidden+6))
            price = jnp.exp(pl.load(lr, (path, day)))
            traded = (next_shares-old_shares)*price
            trade_sign = jnp.where(traded>=0., jnp.asarray(1., prices.dtype), jnp.asarray(-1., prices.dtype))
            chex.assert_shape(trade_sign, ())
            trade_gradient = cot*(1+cost*trade_sign)
            next_share_gradient = gshare+trade_gradient*price
            previous_share_gradient = -trade_gradient*price
            lp_gradient = trade_gradient*traded+pending_lp
            allocation_gradient = next_share_gradient*cap if historical else next_share_gradient*cap/price
            if not historical:
                lp_gradient = lp_gradient-next_share_gradient*next_shares
            logit_gradient = allocation_gradient*allocation*(1-allocation)
            gwo = gwo+logit_gradient*following
            gbo = gbo+logit_gradient
            gh = gh+logit_gradient*wo
            gn = gh*(1-update)*(1-new**2)
            gz = gh*(old_hidden-new)*update*(1-update)
            gr = gn*(hn+bn)*reset*(1-reset)
            ghn = gn*reset
            first_gates = jnp.concatenate((gr, gz))
            last_input = jnp.concatenate((gn, jnp.zeros_like(gn)))
            last_hidden = jnp.concatenate((ghn, jnp.zeros_like(gn)))
            chex.assert_shape((first_gates, last_input, last_hidden), (2*hidden,))
            ig = jnp.concatenate((first_gates, last_input))
            hg = jnp.concatenate((first_gates, last_hidden))
            chex.assert_shape((ig, hg), (rows_size,))
            gwi = gwi+ig[:, None]*x[None, :]
            gwh = gwh+hg[:, None]*old_hidden[None, :]
            gb, gbn = gb+ig, gbn+ghn
            gh = jnp.sum(wh*hg[:, None], axis=0)+gh*update
            gx = jnp.sum(wi*ig[:, None], axis=0)
            chex.assert_shape(gh, (hidden,))
            chex.assert_shape(gx, (4,))
            if historical:
                previous_share_gradient = previous_share_gradient+gx[3]
                lp_gradient = lp_gradient+gx[0]
                previous_lp = gx[1]
            else:
                previous_share_gradient = previous_share_gradient+gx[3]*price/cap
                lp_gradient = lp_gradient+gx[0]+gx[1]+gx[3]*old_shares*price/cap
                previous_lp = -gx[1]
            lp_gradient = lp_gradient+jnp.where(day == 0, previous_lp, 0.)
            pl.store(glr, (path, day), lp_gradient)
            return gh, previous_share_gradient, previous_lp, gwi, gwh, gb, gbn, gwo, gbo

        dtype = prices.dtype
        initial = (jnp.zeros(hidden, dtype), share_gradient, jnp.asarray(0., dtype),
                   jnp.zeros((rows_size, 4), dtype), jnp.zeros((rows_size, hidden), dtype),
                   jnp.zeros(rows_size, dtype), jnp.zeros(hidden, dtype), jnp.zeros(hidden, dtype), jnp.asarray(0., dtype))
        _, _, _, gwi, gwh, gb, gbn, gwo, gbo = jax.lax.fori_loop(0, n_steps, step, initial)
        pl.store(gwir, (path, rows[:, None], features_index[None, :]), gwi, mask=rows[:, None]<3*hidden)
        pl.store(gwhr, (path, rows[:, None], cols[None, :]), gwh, mask=rows[:, None]<3*hidden)
        pl.store(gbr, (path, rows), gb, mask=rows<3*hidden)
        pl.store(gbnr, (path, cols), gbn)
        pl.store(gwor, (path, 0, cols), gwo)
        pl.store(gbor, (path, 0), gbo)

    shapes = tuple(jax.ShapeDtypeStruct((batch, *parameter.shape), parameter.dtype) for parameter in parameters)
    shapes += (jax.ShapeDtypeStruct(prices.shape, prices.dtype),)
    results = pl.pallas_call(kernel, out_shape=shapes, grid=(batch,), name='persistent_hedger_backward')(*parameters, prices, tape, cotangent)
    gradients = tuple(jnp.sum(value, axis=0) for value in results[:-1])
    for gradient, parameter in zip(gradients, parameters):
        chex.assert_equal_shape((gradient, parameter))
    return gradients, results[-1]


losses.defvjp(forward_rule, backward_rule)


class PersistentHedger(eqx.Module):
    """Wrap an existing hedger without changing its parameter values or ordering."""
    model: eqx.Module

    def terminal_losses(self, log_spot, historical=False, cost=.0003, premium=.04, cap=1., strike=1.):
        """Return fused complete-trajectory losses for the selected option contract convention."""
        model = self.model
        cell = model.recurrent
        parameters = (cell.weight_ih, cell.weight_hh, cell.bias, cell.bias_n, model.output.weight, model.output.bias)
        return losses(parameters, log_spot, cost, cap, premium, strike, historical)
