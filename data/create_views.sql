CREATE VIEW temp_signal_detector_eid AS 
SELECT eid FROM sample_assignments_generic_hadron WHERE sample_type = :sample_type;

CREATE VIEW temp_signal_detector_meta AS
SELECT * FROM 
  (temp_signal_detector_eid INNER JOIN event_labels_generic USING (eid))
  INNER JOIN 
  event_weights_generic
  USING (eid);

CREATE VIEW temp_signal_detector_sample AS
SELECT * FROM 
  temp_signal_detector_meta
  INNER JOIN
  candidate_optimized_events_generic
  USING (eid);
