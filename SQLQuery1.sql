select * from pizza_types;

select * from empdata;

SELECT 
    c.COLUMN_NAME,
    c.DATA_TYPE,
    c.IS_NULLABLE,
    c.COLUMN_DEFAULT,
    CASE WHEN k.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END AS IS_PRIMARY_KEY
FROM INFORMATION_SCHEMA.COLUMNS c
LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k 
    ON c.TABLE_NAME = k.TABLE_NAME 
    AND c.COLUMN_NAME = k.COLUMN_NAME
    AND k.CONSTRAINT_NAME IN (
        SELECT CONSTRAINT_NAME 
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
        WHERE CONSTRAINT_TYPE = 'PRIMARY KEY'
    )
WHERE c.TABLE_NAME = 'pizza_types';

-- Create the 'employees' table
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    salary DECIMAL(10, 2),
    project_id INT
);

-- Create the 'project_name' table
CREATE TABLE project_name (
    project_id INT PRIMARY KEY,
    emp_id INT,
    project_name VARCHAR(100),

);

INSERT INTO employees (emp_id, first_name, last_name, salary, project_id) VALUES
(1, 'Alice', 'Johnson', 75000.00, 101),
(2, 'Bob', 'Smith', 62000.00, 102),
(3, 'Carol', 'Taylor', 68000.00, 103),
(4, 'David', 'Brown', 71000.00, 104),
(5, 'Eva', 'Davis', 80000.00, 101),
(6, 'Frank', 'Wilson', 59000.00, 102),
(7, 'Grace', 'Clark', 67000.00, 103),
(8, 'Henry', 'Lewis', 73000.00, 104),
(9, 'Ivy', 'Walker', 76000.00, 101),
(10, 'Jack', 'Hall', 65000.00, 102);


INSERT INTO project_name (project_id, emp_id, project_name) VALUES
(101, 1, 'Apollo'),
(102, 2, 'Gemini'),
(103, 3, 'Orion'),
(104, 4, 'Voyager');


select * from employees;

select SYSTEM_USER;

select @@SERVERNAME;



select * from nukkad_revenue;

select AVG(TOTAL_AMOUNT) as avg_amount from nukkad_revenue where CATEGORY = 'breakfast';

SELECT 
    YEAR(DATE) AS year,
    MONTH(DATE) AS month,
    AVG(total_amount) AS avg_amount
FROM 
    nukkad_revenue
where category = 'breakfast'
GROUP BY 
    YEAR(DATE),
    MONTH(DATE)
ORDER BY
    year,month;
