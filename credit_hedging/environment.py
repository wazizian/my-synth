"""Self-financing credit cashflows with long-only, independently tradable CDS vintages."""
import chex
import jax
import jax.numpy as jnp

from .market import MarketConfig


def book_step(book, action, market, tables, config: MarketConfig):
    """Trade and settle one day; return next book and auditable cashflow/mark records.

    book: PyTree with notional/coupon [batch, cds_days]. At a decision, slot zero
    is empty and reserved for today's issue; slot age a has cds_days-a left.
    action: buy [batch] in face-value units and retain [batch, cds_days] in [0,1].
    Existing contracts may only be reduced. Invalid actions are clipped; total
    protection is capped by config.max_protection. market supplies day (scalar),
    state/next_state, alive/default [batch]. All trades occur before today's
    default is known. Default and maturity cashflows occur at the day's end.
    Optional market['quotes'] supplies bond [batch] and CDS legs
    [batch,cds_days+1], including next-date quotes, instead of grid lookups.
    """
    batch, n_vintages = book['notional'].shape
    chex.assert_shape((book['notional'], book['coupon'], action['retain']), (batch, n_vintages))
    chex.assert_shape(action['buy'], (batch,))
    chex.assert_shape((market['state'], market['next_state'], market['alive'], market['default']), (batch,))
    chex.assert_shape(market['day'], ())
    assert n_vintages == config.cds_days
    dt = 1.0 / config.trading_days
    remaining = config.cds_days - jnp.arange(n_vintages)
    chex.assert_shape(remaining, (n_vintages,))
    state, next_state = market['state'], market['next_state']
    alive = market['alive']
    survive = alive & ~market['default']
    chex.assert_shape(survive, (batch,))
    remaining_bond = config.bond_days - market['day']
    chex.assert_shape(remaining_bond, ())
    final = remaining_bond == 1
    chex.assert_shape(final, ())
    if 'quotes' in market:
        quotes = market['quotes']
        old_marks = quotes['protection'][:, remaining] - book['coupon'] * quotes['annuity'][:, remaining]
        current_bond = quotes['bond']
        next_bond = quotes['next_bond']
        par_spread = quotes['protection'][:, config.cds_days] / quotes['annuity'][:, config.cds_days]
    else:
        old_marks = tables['protection'][remaining[None, :], state[:, None]] - book['coupon'] * tables['annuity'][remaining[None, :], state[:, None]]
        current_bond = tables['bond'][remaining_bond, state]
        next_bond = tables['bond'][remaining_bond-1, next_state]
        par_spread = tables['par_spread'][state]
    chex.assert_shape((current_bond, next_bond, par_spread), (batch,))
    chex.assert_shape(old_marks, (batch, n_vintages))
    old_cds_value = jnp.sum(book['notional'] * old_marks, axis=1)
    chex.assert_shape(old_cds_value, (batch,))
    old_bond_value = jnp.where(alive, current_bond, 0)
    chex.assert_shape(old_bond_value, (batch,))
    retained = book['notional'] * jnp.clip(action['retain'], 0, 1)
    chex.assert_shape(retained, (batch, n_vintages))
    sold = book['notional'] - retained
    chex.assert_shape(sold, (batch, n_vintages))
    room = jnp.maximum(config.max_protection - jnp.sum(retained, axis=1), 0)
    chex.assert_shape(room, (batch,))
    purchased = jnp.where(alive, jnp.clip(action['buy'], 0, room), 0)
    chex.assert_shape(purchased, (batch,))
    holdings = retained.at[:, 0].set(purchased)
    chex.assert_shape(holdings, (batch, n_vintages))
    holdings = jnp.where(alive[:, None], holdings, 0)
    chex.assert_shape(holdings, (batch, n_vintages))
    coupons = book['coupon'].at[:, 0].set(par_spread)
    chex.assert_shape(coupons, (batch, n_vintages))
    sale_cash = jnp.where(alive, jnp.sum(sold * old_marks, axis=1), 0)
    chex.assert_shape(sale_cash, (batch,))
    traded_notional = jnp.where(alive, jnp.sum(sold, axis=1) + purchased, 0)
    chex.assert_shape(traded_notional, (batch,))
    opening_cost = config.transaction_cost * traded_notional
    chex.assert_shape(opening_cost, (batch,))
    # Today's newly issued par CDS has zero upfront mid value. Old coupons never reset.
    premium = dt * jnp.sum(holdings * coupons, axis=1)
    chex.assert_shape(premium, (batch,))
    total_protection = jnp.sum(holdings, axis=1)
    chex.assert_shape(total_protection, (batch,))
    protection_cash = jnp.where(market['default'] & alive, (1-config.recovery)*total_protection, 0)
    chex.assert_shape(protection_cash, (batch,))
    bond_cash = jnp.where(alive & market['default'], config.recovery, jnp.where(survive & final, 1.0, 0.0))
    chex.assert_shape(bond_cash, (batch,))
    # Age after the interval. The last slot matures naturally, without a sale fee.
    next_notional = jnp.concatenate((jnp.zeros((batch, 1)), holdings[:, :-1]), axis=1)
    next_coupon = jnp.concatenate((jnp.zeros((batch, 1)), coupons[:, :-1]), axis=1)
    chex.assert_shape((next_notional, next_coupon), (batch, n_vintages))
    next_notional = jnp.where(survive[:, None], next_notional, 0)
    chex.assert_shape(next_notional, (batch, n_vintages))
    if 'quotes' in market:
        next_marks = quotes['next_protection'][:, remaining] - next_coupon * quotes['next_annuity'][:, remaining]
    else:
        next_marks = tables['protection'][remaining[None, :], next_state[:, None]] - next_coupon * tables['annuity'][remaining[None, :], next_state[:, None]]
    chex.assert_shape(next_marks, (batch, n_vintages))
    next_cds_value = jnp.sum(next_notional * next_marks, axis=1)
    chex.assert_shape(next_cds_value, (batch,))
    final_sale = jnp.where(final, next_cds_value, 0)
    chex.assert_shape(final_sale, (batch,))
    final_cost = jnp.where(final, config.transaction_cost*jnp.sum(next_notional, axis=1), 0)
    chex.assert_shape(final_cost, (batch,))
    next_notional = jnp.where(final, 0, next_notional)
    chex.assert_shape(next_notional, (batch, n_vintages))
    next_coupon = jnp.where(next_notional > 0, next_coupon, 0)
    chex.assert_shape(next_coupon, (batch, n_vintages))
    next_cds_value = jnp.where(final, 0, next_cds_value)
    next_bond_value = jnp.where(survive & ~final, next_bond, 0)
    chex.assert_shape((next_cds_value, next_bond_value), (batch,))
    discount_start = jnp.exp(-config.rate*market['day']*dt)
    discount_end = jnp.exp(-config.rate*(market['day']+1)*dt)
    chex.assert_shape((discount_start, discount_end), ())
    cashflow = discount_start*(sale_cash-opening_cost) + discount_end*(bond_cash+protection_cash-premium+final_sale-final_cost)
    chex.assert_shape(cashflow, (batch,))
    daily_loss = discount_start*(old_bond_value+old_cds_value) - discount_end*(next_bond_value+next_cds_value) - cashflow
    chex.assert_shape(daily_loss, (batch,))
    costs = discount_start*opening_cost + discount_end*final_cost
    chex.assert_shape(costs, (batch,))
    return dict(notional=next_notional, coupon=next_coupon), dict(
        daily_loss=daily_loss, discounted_cashflow=cashflow, discounted_cost=costs,
        buy=purchased, sold=jnp.sum(sold, axis=1), notional=total_protection,
        premium=premium, protection_cash=protection_cash, bond_cash=bond_cash,
        sale_cash=sale_cash, final_sale=final_sale,
        bond_value=next_bond_value, cds_value=next_cds_value)


def rollout(policy, paths, tables, config: MarketConfig, record_inventory=False):
    """Run a policy on immutable paths; return loss [batch] and daily records [batch,time].

    policy is an Equinox recurrent PyTree or None (unhedged reference). No latent
    state or future default enters policy inputs; current quoted par spread and
    inventory marks do. record_inventory adds vintage arrays [batch,time,tenor].
    Loss is initial bond price minus discounted final cashflows for a long bond
    and purchased protection; the funded bond position is not a short liability.
    """
    batch, n_dates = paths['state'].shape
    n_time, n_vintages = n_dates-1, config.cds_days
    assert n_time == config.bond_days
    chex.assert_shape(paths['alive'], (batch, n_time+1))
    chex.assert_shape(paths['default'], (batch, n_time))
    book = dict(notional=jnp.zeros((batch, n_vintages)), coupon=jnp.zeros((batch, n_vintages)))
    chex.assert_shape((book['notional'], book['coupon']), (batch, n_vintages))
    hidden_size = policy.hidden_size if policy is not None else 1
    hidden = jnp.zeros((batch, hidden_size))
    chex.assert_shape(hidden, (batch, hidden_size))
    previous_spread = tables['par_spread'][paths['state'][:, 0]]
    chex.assert_shape(previous_spread, (batch,))
    remaining = config.cds_days - jnp.arange(n_vintages)
    chex.assert_shape(remaining, (n_vintages,))

    def step(carry, day):
        """Apply causal policy and one settlement; input book/hidden/spread and day; output carry/logs."""
        book, hidden, previous_spread = carry
        state = paths['state'][:, day]
        spread = tables['par_spread'][state]
        chex.assert_shape((state, spread), (batch,))
        if policy is None:
            action = dict(buy=jnp.zeros(batch), retain=jnp.ones((batch, n_vintages)))
        else:
            marks = tables['protection'][remaining[None, :], state[:, None]] - book['coupon'] * tables['annuity'][remaining[None, :], state[:, None]]
            chex.assert_shape(marks, (batch, n_vintages))
            # Empty-slot marks are masked: no fictitious coupon-zero instrument input.
            marks = jnp.where(book['notional'] > 0, marks, 0)
            chex.assert_shape(marks, (batch, n_vintages))
            global_features = jnp.stack((100*spread, 100*(spread-previous_spread),
                jnp.full(batch, (n_time-day)/n_time),
                jnp.sum(book['notional'], axis=1)/config.max_protection), axis=1)
            chex.assert_shape(global_features, (batch, 4))
            features = jnp.concatenate((global_features, book['notional']/config.max_protection,
                                        100*book['coupon'], 100*marks), axis=1)
            n_features = 4 + 3*n_vintages
            chex.assert_shape(features, (batch, n_features))
            hidden, action = jax.vmap(policy)(features, hidden)
            chex.assert_shape(hidden, (batch, hidden_size))
        market = dict(day=day, state=state, next_state=paths['state'][:, day+1],
                      alive=paths['alive'][:, day], default=paths['default'][:, day])
        book, records = book_step(book, action, market, tables, config)
        if record_inventory:
            records = dict(records, inventory=book['notional'], coupons=book['coupon'])
        return (book, hidden, spread), records

    days = jnp.arange(n_time)
    chex.assert_shape(days, (n_time,))
    _, records = jax.lax.scan(step, (book, hidden, previous_spread), days)
    records = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), records)
    chex.assert_shape(records['daily_loss'], (batch, n_time))
    initial_price = tables['bond'][n_time, paths['state'][:, 0]]
    chex.assert_shape(initial_price, (batch,))
    cash = jnp.sum(records['discounted_cashflow'], axis=1)
    chex.assert_shape(cash, (batch,))
    loss = initial_price - cash
    chex.assert_shape(loss, (batch,))
    defaulted = jnp.any(paths['default'], axis=1)
    chex.assert_shape(defaulted, (batch,))
    return dict(loss=loss, initial_price=initial_price, defaulted=defaulted, records=records)
