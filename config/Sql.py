import mysql.connector
from Application.Model.Config.Config import Config
configuration = Config()
class Sql:
    selecting = ''
    updating = ''
    deleting = ''
    db_name = ''
    # _construct from oop
    def __init__(self):
        # get the connection of mysql
        self.db = mysql.connector.connect(host=configuration.get_property('dbHost'),
                                          user=configuration.get_property('dbUser'),
                                          passwd=configuration.get_property('dbPassword'),
                                          db=configuration.get_property('dbName'))
        self.cursor = self.db.cursor()
        self.db_name = configuration.get_property('dbName')

    # select only one row from table
    def select_one(self, sql):
        self.cursor.execute(sql)
        self.selecting = self.cursor.fetchone()
        return self.selecting

    # select all row from table
    def select_all(self, sql):
        self.cursor.execute(sql)
        self.selecting = self.cursor.fetchall()
        return self.selecting

    # update field from table
    def update(self, sql):
        self.cursor.execute(sql)
        self.updating = self.db.commit()
        return self.updating

    # delete field from table

    def delete(self, sql):
        self.cursor.execute(sql)
        self.deleting = self.db.commit()
        return self.deleting
