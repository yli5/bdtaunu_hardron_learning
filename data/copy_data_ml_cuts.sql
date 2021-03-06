CREATE TEMPORARY VIEW M AS
SELECT 
  eid, 
  cidx,
  ntracks,
  r2all,
  mmiss2,
  mmiss2prime,
  eextra,
  costhetat,
  tag_lp3,
  tag_cosby,
  tag_costhetadl,
  tag_dmass,
  tag_deltam,
  tag_costhetadsoft,
  tag_softp3magcm,
  sig_hp3,
  sig_cosby,
  sig_costhetadtau,
  sig_vtxb,
  sig_dmass,
  sig_deltam,
  sig_costhetadsoft,
  sig_softp3magcm,
  sig_hmass,
  sig_vtxh,
  cand_score,
  tag_isbdstar,
  sig_isbdstar,
  tag_dmode,
  tag_dstarmode,
  sig_dmode,
  sig_dstarmode,
  tag_l_epid,
  tag_l_mupid,
  weight,
  eventlabel
FROM temp_signal_detector_sample
WHERE r2all <= 0.3
 AND mmiss2 >= 8.0
 AND eextra <= 2.0;
  

\copy (SELECT * FROM M) TO STDOUT WITH CSV HEADER;
