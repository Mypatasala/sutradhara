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

echo "Query 7: Students present in Grade 5"
curl -X POST $URL -H "$HEADER" -d '{"query": "How many students are present in class 5?"}'
echo -e "\n"

echo "Query 8: Total students present in school"
curl -X POST $URL -H "$HEADER" -d '{"query": "How many students are present in school?"}'
echo -e "\n"

echo "Query 9: Student count per course"
curl -X POST $URL -H "$HEADER" -d '{"query": "generate report each class has how many students?"}'
echo -e "\n"

echo "Query 10: Student count per grade"
curl -X POST $URL -H "$HEADER" -d '{"query": "generate report each grade has how many students?"}'
echo -e "\n"

echo "Query 11: List all students in Grade 9"
curl -X POST $URL -H "$HEADER" -d '{"query": "can you list all the students in class 9?"}'
echo -e "\n"
