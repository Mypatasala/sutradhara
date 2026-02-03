import sqlite3
import os

DB_PATH = "school.db"

def init_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Enable window functions (if supported by sqlite version)
    # Most modern sqlite versions support this.
    
    print("Creating tables...")
    
    # Schema definitions
    schema = """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        role TEXT NOT NULL -- 'student', 'teacher', 'admin', 'principal', 'parent'
    );

    CREATE TABLE departments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        head_id INTEGER,
        FOREIGN KEY (head_id) REFERENCES users(id)
    );

    CREATE TABLE students (
        user_id INTEGER PRIMARY KEY,
        grade_level INTEGER,
        parent_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (parent_id) REFERENCES users(id)
    );

    CREATE TABLE teachers (
        user_id INTEGER PRIMARY KEY,
        department_id INTEGER,
        specialization TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (department_id) REFERENCES departments(id)
    );

    CREATE TABLE courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        teacher_id INTEGER,
        FOREIGN KEY (teacher_id) REFERENCES teachers(user_id)
    );

    CREATE TABLE enrollments (
        student_id INTEGER,
        course_id INTEGER,
        enrollment_date DATE,
        PRIMARY KEY (student_id, course_id),
        FOREIGN KEY (student_id) REFERENCES students(user_id),
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );

    CREATE TABLE timetable (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER,
        day_of_week TEXT,
        start_time TIME,
        end_time TIME,
        room_number TEXT,
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );

    CREATE TABLE assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER,
        title TEXT,
        description TEXT,
        due_date DATE,
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );

    CREATE TABLE exams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER,
        title TEXT,
        exam_date DATE,
        type TEXT, -- 'midterm', 'final', 'quiz'
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );

    CREATE TABLE report_cards (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER,
        grade REAL,
        remarks TEXT,
        exam_period TEXT,
        FOREIGN KEY (student_id) REFERENCES students(user_id),
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );

    CREATE TABLE attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        date DATE,
        status TEXT, -- 'present', 'absent', 'late'
        FOREIGN KEY (student_id) REFERENCES students(user_id)
    );

    CREATE TABLE library_books (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        author TEXT,
        isbn TEXT UNIQUE,
        category TEXT
    );

    CREATE TABLE library_borrows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_id INTEGER,
        student_id INTEGER,
        borrow_date DATE,
        due_date DATE,
        return_date DATE,
        status TEXT, -- 'borrowed', 'returned', 'overdue'
        FOREIGN KEY (book_id) REFERENCES library_books(id),
        FOREIGN KEY (student_id) REFERENCES students(user_id)
    );

    CREATE TABLE fee_payments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        amount REAL,
        payment_date DATE,
        payment_method TEXT,
        status TEXT, -- 'paid', 'pending', 'overdue'
        FOREIGN KEY (student_id) REFERENCES students(user_id)
    );

    CREATE TABLE clubs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        teacher_id INTEGER,
        FOREIGN KEY (teacher_id) REFERENCES teachers(user_id)
    );

    CREATE TABLE club_memberships (
        student_id INTEGER,
        club_id INTEGER,
        join_date DATE,
        PRIMARY KEY (student_id, club_id),
        FOREIGN KEY (student_id) REFERENCES students(user_id),
        FOREIGN KEY (club_id) REFERENCES clubs(id)
    );
    """
    
    cursor.executescript(schema)
    
    print("Ingesting massive data...")
    from faker import Faker
    import random
    from datetime import datetime, timedelta

    fake = Faker()
    
    # 1. Departments
    depts = ['Science', 'Math', 'Arts', 'Physical Education', 'History', 'Computer Science', 'Languages']
    dept_ids = []
    for d in depts:
        cursor.execute("INSERT INTO departments (name) VALUES (?)", (d,))
        dept_ids.append(cursor.lastrowid)

    # 2. Users & Roles
    # Principals (2)
    principal_ids = []
    for _ in range(2):
        name = fake.name()
        email = fake.unique.email()
        cursor.execute("INSERT INTO users (name, email, role) VALUES (?, ?, 'principal')", (name, email))
        principal_ids.append(cursor.lastrowid)
    
    # Admins (10)
    for _ in range(10):
        name = fake.name()
        email = fake.unique.email()
        cursor.execute("INSERT INTO users (name, email, role) VALUES (?, ?, 'admin')", (name, email))
    
    # Teachers (100)
    teacher_user_ids = []
    specializations = ['Physics', 'Calculus', 'Literature', 'Chemistry', 'Biology', 'History', 'Web Dev', 'AI', 'French', 'Spanish']
    for _ in range(100):
        name = fake.name()
        email = fake.unique.email()
        cursor.execute("INSERT INTO users (name, email, role) VALUES (?, ?, 'teacher')", (name, email))
        user_id = cursor.lastrowid
        teacher_user_ids.append(user_id)
        
        dept_id = random.choice(dept_ids)
        spec = random.choice(specializations)
        cursor.execute("INSERT INTO teachers (user_id, department_id, specialization) VALUES (?, ?, ?)", (user_id, dept_id, spec))

    # Parents (450)
    parent_ids = []
    for _ in range(450):
        name = fake.name()
        email = fake.unique.email()
        cursor.execute("INSERT INTO users (name, email, role) VALUES (?, ?, 'parent')", (name, email))
        parent_ids.append(cursor.lastrowid)

    # Students (500)
    student_ids = []
    for i in range(500):
        name = fake.name()
        email = fake.unique.email()
        parent_id = random.choice(parent_ids)
        grade_level = random.randint(9, 12)
        
        cursor.execute("INSERT INTO users (name, email, role) VALUES (?, ?, 'student')", (name, email))
        user_id = cursor.lastrowid
        cursor.execute("INSERT INTO students (user_id, grade_level, parent_id) VALUES (?, ?, ?)", (user_id, grade_level, parent_id))
        student_ids.append(user_id)

    # 3. Courses (50)
    course_ids = []
    for _ in range(50):
        spec = random.choice(specializations)
        name = f"{spec} {random.randint(101, 404)}"
        teacher_id = random.choice(teacher_user_ids)
        cursor.execute("INSERT INTO courses (name, description, teacher_id) VALUES (?, ?, ?)", (name, fake.sentence(), teacher_id))
        course_ids.append(cursor.lastrowid)

    # 4. Enrollments (Each student in 4-6 courses)
    enrollment_data = []
    report_card_data = []
    for s_id in student_ids:
        num_courses = random.randint(4, 6)
        selected_courses = random.sample(course_ids, num_courses)
        for c_id in selected_courses:
            enrollment_data.append((s_id, c_id, '2024-01-15'))
            # Initial grade
            grade = round(random.uniform(55, 100), 2)
            report_card_data.append((s_id, c_id, grade, 'Semester 1'))

    cursor.executemany("INSERT INTO enrollments (student_id, course_id, enrollment_date) VALUES (?, ?, ?)", enrollment_data)
    cursor.executemany("INSERT INTO report_cards (student_id, course_id, grade, exam_period) VALUES (?, ?, ?, ?)", report_card_data)

    # 5. Assignments (2 per course)
    assignment_data = []
    for c_id in course_ids:
        assignment_data.append((c_id, "Homework 1", "Basic review", "2024-02-15"))
        assignment_data.append((c_id, "Project 1", "Deep dive", "2024-03-01"))
    cursor.executemany("INSERT INTO assignments (course_id, title, description, due_date) VALUES (?, ?, ?, ?)", assignment_data)

    # 6. Attendance (~15,000 records: 30 days for 500 students)
    print("Generating 15,000 attendance records...")
    attendance_data = []
    start_date = datetime(2024, 2, 1)
    for student_id in student_ids:
        for day in range(30):
            current_date = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')
            # Using Title Case for status to match LLM expectations
            status = random.choices(['Present', 'Absent', 'Late'], weights=[0.9, 0.05, 0.05])[0]
            attendance_data.append((student_id, current_date, status))
    
    # Use batching for the huge attendance insert
    batch_size = 1000
    for i in range(0, len(attendance_data), batch_size):
        cursor.executemany("INSERT INTO attendance (student_id, date, status) VALUES (?, ?, ?)", attendance_data[i:i+batch_size])

    # 7. Library Data
    # Adding 'Science' as a category for some books to match common queries
    lib_categories = specializations + ['Science', 'Technology', 'Fiction']
    books = [(fake.catch_phrase(), fake.name(), fake.isbn13(), random.choice(lib_categories)) for _ in range(100)]
    cursor.executemany("INSERT INTO library_books (title, author, isbn, category) VALUES (?, ?, ?, ?)", books)
    
    # 50 random borrows
    borrow_data = []
    for _ in range(50):
        book_id = random.randint(1, 100)
        s_id = random.choice(student_ids)
        borrow_data.append((book_id, s_id, '2024-02-01', '2024-02-15', 'Borrowed'))
    cursor.executemany("INSERT INTO library_borrows (book_id, student_id, borrow_date, due_date, status) VALUES (?, ?, ?, ?, ?)", borrow_data)

    conn.commit()
    conn.close()
    print(f"Database {DB_PATH} expanded successfully with 500 students, 100 teachers, and 15,000+ attendance records.")

if __name__ == "__main__":
    init_db()
