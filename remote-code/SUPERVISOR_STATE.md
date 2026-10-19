# Completed campaign

Post-campaign user-authorized cleanup completed 2026-10-18T08:59:12.985351+00:00. Git/code and all canonical metrics retained; datasets/checkpoints/history/caches/worker exports removed. Not resumable from deleted states. Historical audits remain evidence of pre-cleanup success. See results/cleanup_manifest.json.

Task: execute /home/jump/plan.md. All required experiment, evaluation, export, reporting and shutdown work is complete.
Completed UTC: 2026-10-18T02:29:46.975528+00:00. Campaign start21:54UTC2026-10-17; duration4h 35m 46s (limit8hours).
Provider-reported costUSD3.681 asof2026-10-18T02:28:09.203212+00:00 (rounded; maysettleaftershutdown). Conservativecontrol-plane totalUSD12.579; notadditionalcharges. LimitUSD25. Zeroactivecampaigninstances, zerounreportedIDs, zerooutstandingreservations. Nostandalonevolumescreated.

## Scientific outcomes and artifacts

Exactlyseven scientificarms: baseline,hard_reference,dpg_hard,soar_hard,selfplay,dpg_mixed,soar_mixed. Eachselectedlineagecompleted2700active seconds, one seed20261017. OptionalgeneratorLR.01 uniformlyomitted bystorage/exportreserveproof, notoutcomes.
Selectionfrozen01:54:09UTC2026-10-18. All169checkpoints across21retainedhedgerlineages evaluated on3finaldomains x8192episodes. 1191successfuleventrecords checkedagainstsavedarrays. GPUrawvalidation andartifactcompleteness passed. 34figuresretained.
Baselinehaslowestbase/easy andbalanced MSE. SOARmixedlowestobservedhard MSE, butsmallgapandworsebase/easy. DPGmixedbetterthanDPGhard,butbothtrailbaselineinselected-testMSE. Selfplaymovedcurriculumstronglywithoutbroadgain. Allsingle-seedexploratory; fullreporthaspricing/telemetry/costlimitations.

Startatresults/report.md, results/final_metrics.csv, results/all_trials.csv, results/campaign_signoff.json. results/provenance separatescheckpoint/evaluatorsourcebuildsandproviderworkercosts. Do notsumrepeateddomaincostfields.

## Validation and recovery

20/20GPUtests,7smokes,7productionprofiles,actualselfplayresume/SIGTERMrecoverypassed. Pricingtable/fallbackvalidationpassed; final7selectedpairedpricingdiagnostic32monitor/domaincomplete. Small-subsetMSEchanges0.022%-0.978%, gapchanges0.083%-20.13%, nosubsetsignreversals. Thisdoesnotcertifytinyfull-testgaps/tails.
Trainingpricingrangetelemetryissampledblocks, notcumulativetrainingtotals. Everyblockchecksfinitepricesbeforelearning. Totaltrainingfallbackfrequency/maxexcursionwerenotrecorded; seeresults/telemetry_scope_review.json.
Recoveryauditpassed169checkpoints,610229historyreferences,61233uniquepayloads. Historicalworkingfiles577across12snapshotsrestoredfromverifiedGitobjectswithoutoverwritingexistingfiles. Originalfailedauditandmaterializationevidencepreservedexternally;passingcopiesinresults.

## Operational closure

Alltraining/evaluation/reportworkers exportedanddestroyed. Staleofferscreatednoinstancesandincurredzerorentalcost. Extensions0 andtests0/2 replacementsreconciledwithguardedhelpers; originalfailedattempts/exactreplacementprovenance retainedinqueue/state.json.
Allcompleteddiagnostic/screen/extension/testhandofftarballsverifiedandthinned; canonicalscientificbytes retained. Thinmetadata/recoveryinventoriesin/home/jump/campaign-control/thin_queue_archives. Reporthandofftarballretained; nofurthercleanupneeded.
Scientificsource frozenchecks passedbeforeclosure; onlyfinaldocumentation/evidenceadded. No numericalworkranonthesupervisor.
No work remains. Do notresumecontroller orrentanotherGPU for thiscompletedcampaign unlessuserrequestsnewwork.
