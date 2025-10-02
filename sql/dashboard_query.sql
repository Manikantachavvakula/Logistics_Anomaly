
-- Main Dashboard Query
-- Use this in Power BI to load shipment data

SELECT 
    shipment_id,
    timestamp,
    carrier_name,
    route,
    service_level,
    shipment_date,
    planned_delivery,
    actual_delivery,
    delay_minutes,
    sla_minutes,
    sla_breach_minutes,
    disruption_cost,
    weather_cost,
    congestion_cost,
    customs_cost,
    total_additional_cost,
    cargo_value,
    weight_kg,
    distance_km,
    stops_count,
    carrier_otd_history,
    weather_delay_minutes,
    port_congestion_minutes,
    customs_delay_minutes,
    risk_classification,
    month,
    quarter,
    is_weekend,
    route_complexity,
    -- Derived fields
    CASE 
        WHEN sla_breach_minutes > 0 THEN 'Breach'
        ELSE 'On-Time'
    END as sla_status,
    CASE 
        WHEN delay_minutes > 2880 THEN 'Critical'
        WHEN delay_minutes > 1440 THEN 'Major'
        WHEN delay_minutes > 0 THEN 'Minor'
        ELSE 'On-Time'
    END as delay_category
FROM shipments
ORDER BY shipment_date DESC;
