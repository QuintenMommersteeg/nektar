import sqlite3, datetime

con = sqlite3.connect("demo.db")
cur = con.cursor()

cur.executescript("""
DROP TABLE IF EXISTS appointments;
DROP TABLE IF EXISTS work_orders;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT NOT NULL
);

CREATE TABLE work_orders(
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  title TEXT,
  description TEXT,
  status TEXT,
  created_at TEXT,
  FOREIGN KEY(customer_id) REFERENCES customers(id)
);

CREATE TABLE appointments(
  id INTEGER PRIMARY KEY,
  customer_id INTEGER,
  start_ts TEXT,
  end_ts TEXT,
  resource TEXT,
  FOREIGN KEY(customer_id) REFERENCES customers(id)
);
""")

customers = [
  ("Jan Jansen","jan@example.com"),
  ("Piet Pieters","piet@example.com"),
  ("Sanne Smit","sanne@example.com"),
]
cur.executemany("INSERT INTO customers(name,email) VALUES(?,?)", customers)

now = datetime.datetime(2025,10,28,12,0,0)
work_orders = [
  (1,"Lekke band","VW Transporter links-achter","planned",(now - datetime.timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")),
  (2,"Jaarlijkse onderhoud","Inspectie + olie","done",(now - datetime.timedelta(days=3)).strftime("%Y-%m-%d %H:%M")),
  (3,"Remmen controleren","Piepend geluid","open",(now - datetime.timedelta(days=10)).strftime("%Y-%m-%d %H:%M")),
]
cur.executemany("INSERT INTO work_orders(customer_id,title,description,status,created_at) VALUES(?,?,?,?,?)", work_orders)

appointments = [
  (2,(now + datetime.timedelta(days=1,hours=3)).strftime("%Y-%m-%d %H:%M"), (now + datetime.timedelta(days=1,hours=4)).strftime("%Y-%m-%d %H:%M"), "Monteur-1"),
  (1,(now + datetime.timedelta(days=2,hours=2)).strftime("%Y-%m-%d %H:%M"), (now + datetime.timedelta(days=2,hours=3)).strftime("%Y-%m-%d %H:%M"), "Monteur-2"),
]
cur.executemany("INSERT INTO appointments(customer_id,start_ts,end_ts,resource) VALUES(?,?,?,?)", appointments)

con.commit(); con.close()
print("demo.db aangemaakt met voorbeelddata.")
