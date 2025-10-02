
-- Route Analysis
-- Analyze performance by route

SELECT 
    route,
    COUNT(*) as shipment_count,
    ROUND(AVG(distance_km), 2) as avg_distance_km,
    ROUND(AVG(delay_minutes), 2) as avg_delay_minutes,
    ROUND(AVG(route_complexity), 3) as avg_complexity,
    SUM(CASE WHEN sla_breach_minutes > 0 THEN 1 ELSE 0 END) as breaches,
    ROUND(SUM(total_additional_cost), 2) as total_cost
FROM shipments
GROUP BY route
ORDER BY total_cost DESC;
