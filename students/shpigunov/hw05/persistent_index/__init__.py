import sqlite3


class PersistentKeyValueStorage:

    def __init__(self, db_filename='index.sqlite', table_name='key_value_1'):
        """Initialize persistent index with specified parameters"""

        self.conn = sqlite3.connect(db_filename)
        self.c = self.conn.cursor()

        self.table_name = table_name

        self.c.execute(f"""CREATE TABLE IF NOT EXISTS {self.table_name}
                           (key TEXT, value INTEGER);""")
    
    def insert(self, key, value):
        """Insert a new key:value pair into index"""

        self.c.execute(f"""INSERT INTO {self.table_name} (key, value) VALUES ('{key}', {value});""")

    def get_all(self):
        
        res = set()
        self.c.execute(f"""SELECT * FROM {self.table_name};""")
        
        for item in self.c.fetchall():
            res.add((item[0], item[1]))
        return res        
        
        
    def save_state(self):
        """Commit changes to the index into database"""
        self.conn.commit()
        
    def close(self, save_state=True):
        """Close connection to the database, saving state by default"""

        if save_state:
            self.save_state()

        self.conn.close()
        

class PersistentIndex:
    """This is a persistent structure to store data for all visited and unvisited links"""

    def __init__(self, db_filename='index.sqlite', table_name='index1'):
        """Initialize persistent index with specified parameters"""

        self.conn = sqlite3.connect(db_filename)
        self.c = self.conn.cursor()

        self.table_name = table_name

        self.c.execute(f"""CREATE TABLE IF NOT EXISTS {self.table_name}
                           (item text, status INTEGER);""")

    def load_state(self):
        pass

    def save_state(self):
        """Commit changes to the index into database"""
        self.conn.commit()

    def mark_as_visited(self, s):
        """Mark an entry with name `s` as visited"""

        self.c.execute(f"""update my_index
                            set status = 1
                            where item = '{s}';""")

    def mark_as_unvisited(self, s):
        """Mark an entry with name `s` as unvisited"""

        self.c.execute(f"""update my_index
                            set status = 0
                            where item = '{s}';""")

    def mark_as_error(self, s):
        """Mark an entry with name `s` as containing an error"""

        self.c.execute(f"""update my_index
                            set status = -1
                            where item = '{s}';""")

    def add_unvisited(self, s):
        """Add a new unvisited entry to the index"""

        self.c.execute(f"""INSERT INTO {self.table_name} (item, status) VALUES ('{s}', 0);""")

    def get_unvisited(self):
        """Get a list of all unvisited entries in the index"""

        unvisited = set()
        self.c.execute(f"""SELECT * FROM {self.table_name} WHERE status=0;""")
        for item in self.c.fetchall():
            unvisited.add(item[0])
        return unvisited

    def get_visited(self):
        """Get a list of all visited entries in the index"""

        unvisited = set()
        self.c.execute(f"""SELECT * FROM {self.table_name} WHERE status=1;""")
        for item in self.c.fetchall():
            unvisited.add(item[0])
        return unvisited

    def get_errors(self):
        """Get a list of all entries in the index marked as errors"""

        unvisited = set()
        self.c.execute(f"""SELECT * FROM {self.table_name} WHERE status=-1;""")
        for item in self.c.fetchall():
            unvisited.add(item[0])
        return unvisited

    def update_unvisited(self, new):
        """Wrapper to add multiple items to the index as unvisited from iterable `new`"""

        for item in new:
            self.add_unvisited(item)

    # TODO: to optimize execution, move is_visited and is_unvisited database-side
    # meaning: select 1 entry and check its status, rather than select all with status
    # and check for entry.

    def is_visited(self, s):
        if s in self.get_visited():
            return True
        else:
            return False

    def is_unvisited(self, s):
        if s in self.get_unvisited():
            return True
        else:
            return False

    def wipe(self):
        """Completely and irreversibly wipes the entire index"""

        self.c.execute(f"""DELETE FROM {self.table_name};""")

    def close(self, save_state=True):
        """Close connection to the database, saving state by default"""

        if save_state:
            self.save_state()

        self.conn.close()

