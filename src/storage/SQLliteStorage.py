'''
Created on Nov 14, 2017

@author: abhijit.tomar
'''

import sqlite3 as sql
from sqlite3 import Error
from constants import db_defaults
class CNNSQL(object):
    '''
    Class for storing model data in sqlite
    '''

    def __init__(self, db_name=db_defaults.DB_NAME, table_name=db_defaults.TABLE_NAME):
        
        self.db_name = db_name
        self.table_name = table_name
        
        try:
            conn = sql.connect(self.db_name)
            print ("Opened database successfully");
            conn.execute('''CREATE TABLE IF NOT EXISTS {} (
                        split REAL, 
                        epochs INTEGER, 
                        imageSize INTEGER,
                        learningRate REAL,
                        name TEXT)'''.format(self.table_name))
            print ("Table created successfully");
            cur = conn.cursor()
            cur.execute('select * from {}'.format(self.table_name))
            print([description[0] for description in cur.description])
            conn.close()
        except Error as e:
            print(e)
            
    def add_new_model(self, model_attributes):
        
        with sql.connect(self.db_name) as con:
            
            try:
                cur = con.cursor()
                cur.execute("SELECT * FROM {} WHERE (name=?)".format(self.table_name), (model_attributes['name'],))
                entry = cur.fetchone()
    
                if entry is None:
                    cur.execute('''
                            INSERT INTO {} (
                            split, 
                            epochs, 
                            imageSize,
                            learningRate,
                            name) VALUES (?,?,?,?,?)'''.format(self.table_name),
                            (
                            model_attributes['split'],
                            model_attributes['epochs'],
                            model_attributes['imageSize'],
                            model_attributes['learningRate'],
                            model_attributes['name']))
                    print ('New model {} added'.format(model_attributes['name']))
                else:
                    print ('{} already exists. Will be updated'.format(model_attributes['name']))
                    cur.execute('''
                    UPDATE {}
                    SET 
                        split = ?, 
                        epochs = ?,
                        imageSize = ?,
                        learningRate = ?
                     WHERE    
                         name = ?
                    '''.format(self.table_name),
                    (model_attributes['split'],
                    model_attributes['epochs'],
                    model_attributes['imageSize'],
                    model_attributes['learningRate'],
                    model_attributes['name']))
                con.commit()
            except Error as e:
                con.rollback()
                print(e)
                
    def get_names(self):
        
        name_list = []
        with sql.connect(self.db_name) as con:
            try:
                cur = con.cursor()
                cur.execute('''
                    SELECT name from {}
                '''.format(self.table_name))
                for row in cur.fetchall():
                    name_list.append(row[0])
            except Error as e:
                con.rollback()
                print(e)
        return name_list
    
    def get_attributes(self, modelName):
        model_attributes = {}
        with sql.connect(self.db_name) as con:
            try:
                cur = con.cursor()
                cur.execute('''
                    SELECT * from {}
                '''.format(self.table_name))
                for row in cur.fetchall():
                    for i , col_tup in enumerate(cur.description):
                        model_attributes[col_tup[0]] = row[i]
            except Error as e:
                con.rollback()
                print(e)
        return model_attributes
    
    def drop_table(self):
        
        with sql.connect(self.db_name) as con:
            try:
                cur = con.cursor()
                cur.execute('DROP TABLE IF EXISTS {}'.format(self.table_name))
            except Error as e:
                con.rollback()
                print(e)
            