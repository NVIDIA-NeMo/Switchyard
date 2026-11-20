DELETE FROM dashboards WHERE id='sy-cost-dash';
DELETE FROM dashboard_widgets WHERE id LIKE 'sy-w%';
INSERT INTO dashboard_widgets (id, project_id, name, description, view, dimensions, metrics, filters, chart_type, chart_config, min_version) VALUES
('sy-w1','switchyard','Cost by model: actual vs predicted','Real model cost vs predicted:opus-4-8 (forecast band midpoint)','OBSERVATIONS',
 '[{"field":"providedModelName"}]','[{"agg":"sum","measure":"totalCost"}]','[]','VERTICAL_BAR','{"type":"VERTICAL_BAR","row_limit":20}',2),
('sy-w2','switchyard','Cost over time by model: actual vs predicted','','OBSERVATIONS',
 '[{"field":"providedModelName"}]','[{"agg":"sum","measure":"totalCost"}]','[]','LINE_TIME_SERIES','{"type":"LINE_TIME_SERIES"}',2),
('sy-w3','switchyard','Next model call: predicted cost band vs actual','pred low/mid/high (USD) overlaid with actual call cost','SCORES_NUMERIC',
 '[{"field":"name"}]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_next_call_cost_low","pred_next_call_cost_mid","pred_next_call_cost_high","actual_call_cost"]}]',
 'LINE_TIME_SERIES','{"type":"LINE_TIME_SERIES"}',2),
('sy-w4','switchyard','Task cost: predicted total vs actual so far','pred_total = actual so far + predicted remaining','SCORES_NUMERIC',
 '[{"field":"name"}]','[{"agg":"max","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_total_task_cost","actual_cost_so_far","pred_remaining_task_cost"]}]',
 'LINE_TIME_SERIES','{"type":"LINE_TIME_SERIES"}',2),
('sy-w5','switchyard','Predicted remaining model calls','','SCORES_NUMERIC',
 '[{"field":"name"}]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_remaining_calls"]}]',
 'LINE_TIME_SERIES','{"type":"LINE_TIME_SERIES"}',2),
('sy-w6','switchyard','Next-call band hit rate','share of calls whose actual cost fell inside the predicted band','SCORES_NUMERIC',
 '[]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["next_call_in_band"]}]',
 'NUMBER','{"type":"NUMBER"}',2),
('sy-w7','switchyard','Total cost by model','','OBSERVATIONS',
 '[{"field":"providedModelName"}]','[{"agg":"sum","measure":"totalCost"}]','[]','PIE','{"type":"PIE","row_limit":20}',2);
INSERT INTO dashboards (id, project_id, name, description, definition, filters) VALUES
('sy-cost-dash','switchyard','Switchyard cost: actual vs predicted','Forecaster token bands priced at Opus 4.8 rates, overlaid with actual cost',
 '{"widgets":[
  {"id":"l1","type":"widget","widgetId":"sy-w1","x":0,"y":0,"x_size":6,"y_size":5},
  {"id":"l2","type":"widget","widgetId":"sy-w2","x":6,"y":0,"x_size":6,"y_size":5},
  {"id":"l3","type":"widget","widgetId":"sy-w3","x":0,"y":5,"x_size":8,"y_size":6},
  {"id":"l6","type":"widget","widgetId":"sy-w6","x":8,"y":5,"x_size":4,"y_size":3},
  {"id":"l7","type":"widget","widgetId":"sy-w7","x":8,"y":8,"x_size":4,"y_size":3},
  {"id":"l4","type":"widget","widgetId":"sy-w4","x":0,"y":11,"x_size":8,"y_size":6},
  {"id":"l5","type":"widget","widgetId":"sy-w5","x":8,"y":11,"x_size":4,"y_size":6}]}','[]');
DELETE FROM dashboard_widgets WHERE id IN ('sy-w8','sy-w9','sy-w10');
INSERT INTO dashboard_widgets (id, project_id, name, description, view, dimensions, metrics, filters, chart_type, chart_config, min_version) VALUES
('sy-w8','switchyard','Per call: next-call predicted band vs actual cost','one group per model call (trace name)','SCORES_NUMERIC',
 '[{"field":"traceName"},{"field":"name"}]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_next_call_cost_low","pred_next_call_cost_mid","pred_next_call_cost_high","actual_call_cost"]}]',
 'VERTICAL_BAR','{"type":"VERTICAL_BAR","row_limit":100}',2),
('sy-w9','switchyard','Per call: task cost, predicted total vs actual so far','','SCORES_NUMERIC',
 '[{"field":"traceName"},{"field":"name"}]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_total_task_cost","actual_cost_so_far","pred_remaining_task_cost"]}]',
 'VERTICAL_BAR','{"type":"VERTICAL_BAR","row_limit":100}',2),
('sy-w10','switchyard','Per call table: all predictions vs actuals','','SCORES_NUMERIC',
 '[{"field":"traceName"},{"field":"name"}]','[{"agg":"avg","measure":"value"}]',
 '[{"type":"stringOptions","column":"name","operator":"any of","value":["pred_next_call_cost_low","pred_next_call_cost_mid","pred_next_call_cost_high","actual_call_cost","pred_remaining_calls","pred_remaining_task_cost","pred_total_task_cost","actual_cost_so_far","next_call_in_band"]}]',
 'PIVOT_TABLE','{"type":"PIVOT_TABLE","row_limit":200}',2);
UPDATE dashboards SET definition = jsonb_set(definition::jsonb, '{widgets}', (definition::jsonb->'widgets') || '[
 {"id":"l8","type":"widget","widgetId":"sy-w8","x":0,"y":17,"x_size":12,"y_size":6},
 {"id":"l9","type":"widget","widgetId":"sy-w9","x":0,"y":23,"x_size":12,"y_size":6},
 {"id":"l10","type":"widget","widgetId":"sy-w10","x":0,"y":29,"x_size":12,"y_size":7}]'::jsonb)
WHERE id='sy-cost-dash' AND NOT (definition::text LIKE '%sy-w8%');
DELETE FROM dashboard_widgets WHERE id IN ('sy-w11','sy-w12','sy-w13','sy-w14');
INSERT INTO dashboard_widgets (id, project_id, name, description, view, dimensions, metrics, filters, chart_type, chart_config, min_version)
SELECT v.id, 'switchyard', v.name, v.descr, 'SCORES_NUMERIC', '[{"field":"traceName"}]', '[{"agg":"avg","measure":"value"}]',
  ('[{"type":"stringOptions","column":"name","operator":"any of","value":["' || v.score || '"]}]')::json,
  'VERTICAL_BAR', '{"type":"VERTICAL_BAR","row_limit":100}', 2
FROM (VALUES
 ('sy-w11','Next call: ACTUAL cost per call','actual_call_cost','pair with the predicted chart to the right'),
 ('sy-w12','Next call: PREDICTED cost per call (band mid)','pred_next_call_cost_mid','pair with the actual chart to the left'),
 ('sy-w13','Task: ACTUAL cost so far, per call','actual_cost_so_far','pair with the predicted total to the right'),
 ('sy-w14','Task: PREDICTED total cost, per call','pred_total_task_cost','actual so far + predicted remaining, at each call')
) AS v(id, name, score, descr);
UPDATE dashboards SET definition = jsonb_set(definition::jsonb, '{widgets}', (definition::jsonb->'widgets') || '[
 {"id":"l11","type":"widget","widgetId":"sy-w11","x":0,"y":36,"x_size":6,"y_size":6},
 {"id":"l12","type":"widget","widgetId":"sy-w12","x":6,"y":36,"x_size":6,"y_size":6},
 {"id":"l13","type":"widget","widgetId":"sy-w13","x":0,"y":42,"x_size":6,"y_size":6},
 {"id":"l14","type":"widget","widgetId":"sy-w14","x":6,"y":42,"x_size":6,"y_size":6}]'::jsonb)
WHERE id='sy-cost-dash' AND NOT (definition::text LIKE '%sy-w11%');
-- per-call widgets render as tables (Langfuse bar charts support only one dimension)
UPDATE dashboard_widgets SET chart_type='PIVOT_TABLE', chart_config='{"type":"PIVOT_TABLE","row_limit":200}' WHERE id IN ('sy-w8','sy-w9');
