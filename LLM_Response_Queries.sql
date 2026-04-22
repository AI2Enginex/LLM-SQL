
--- LLM Query Result
SELECT   T2.category,   SUM(T1.quantity) AS total_quantity 
FROM order_details AS T1 JOIN pizzas AS T3 ON T1.pizza_id = T3.pizza_id JOIN pizza_types AS T2   
ON T3.pizza_type_id = T2.pizza_type_id GROUP BY   T2.category;

---- LLM Query Result (Filter by Date)
SELECT   T3.category,   SUM(T1.quantity) AS total_quantity FROM order_details AS T1 JOIN orders AS T2   
ON T1.order_id = T2.order_id JOIN pizzas AS T4   ON T1.pizza_id = T4.pizza_id JOIN pizza_types AS T3   
ON T4.pizza_type_id = T3.pizza_type_id WHERE   YEAR(T2.date) = 2015 AND MONTH(T2.date) = 1 AND DAY(T2.date) = 27 GROUP BY   T3.category;

select * from orders;