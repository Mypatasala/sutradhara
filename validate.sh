#!/bin/bash

# Configuration
URL="http://localhost:8001/api/v1/ask"
HEADER="Content-Type: application/json"

echo "Query 1: Department with most teachers"
curl -X POST $URL -H "$HEADER" -d '{"query": "Which department has the most teachers?"}'
echo -e "\n"

echo "Query 2: Students who borrowed Science books"
curl -X POST $URL -H "$HEADER" -d '{"query": "List the names of students who have borrowed Science books."}'
echo -e "\n"

echo "Query 3: Average grade for Grade 10 students"
curl -X POST $URL -H "$HEADER" -d '{"query": "Calculate the average grade for all students in Grade 10."}'
echo -e "\n"

echo "Query 4: Attendance percentage for last 30 days"
curl -X POST $URL -H "$HEADER" -d '{"query": "Show the attendance percentage for the last 30 days grouped by grade level."}'
echo -e "\n"

echo "Query 5: Total fees from Robotics Club members"
curl -X POST $URL -H "$HEADER" -d '{"query": "How much total fees have been collected from members of the Robotics Club?"}'
echo -e "\n"

echo "Query 6: Top 5 students with most absences"
curl -X POST $URL -H "$HEADER" -d '{"query": "Show the top 5 students with the most absences using lowercase status check."}'
echo -e "\n"
