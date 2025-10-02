
-- Carrier Performance Analysis
-- Use this to analyze carrier metrics

SELECT 
    carrier_name,
    COUNT(*) as total_shipments,
    ROUND(AVG(delay_minutes), 2) as avg_delay_minutes,
    ROUND(AVG(carrier_otd_history), 2) as avg_otd_score,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as total_breaches,
    ROUND(100.0 * SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as breach_rate_pct,
    ROUND(SUM(total_additional_cost), 2) as total_cost_impact,
    ROUND(AVG(total_additional_cost), 2) as avg_cost_per_shipment
FROM shipments
GROUP BY carrier_name
ORDER BY breach_rate_pct DESC;
