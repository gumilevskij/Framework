# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:59:56 2018

@author: agoumilevski
"""
import os
import sys
import sqlite3
from sqlite3 import Error
import argparse


def create_sqlitefile(dbfilename, variable_names):
    """
    Create Python sqlite database file.
    
    Parameters:
        :param dbfilename: Path to database file.
        :type dbfilename: str.
        :param variable_names: Variables names.
        :type variable_names: List.
        :return: Connection object or None
    """
    if not os.path.isfile(dbfilename):
        print('Creating new dbfile: ' + dbfilename)
    else:
        #print('File and table already exists, deleting it and creating new dbfile: ' + dbfilename)
        os.remove(dbfilename) 

    txt = ["'" + var + "'" for var in variable_names]
    txt = " TEXT ,".join(txt) + " TEXT"
    
    conn = sqlite3.connect(dbfilename)
    cur = conn.cursor()

    # Create table
    query = 'CREATE TABLE DATA (' + txt + ')'
    try:
        cur.execute(query)
        conn.commit()
        return conn
    except Error as e:
        print(e,dbfilename)
        
    return None


def insert_values(conn, values, dates = None):
    """
    Take a db connection and an array of values and inserts these values into sqlite file.
    
    Parameters:
        :param conn: sqlite connection object.
        :type conn: Connection.
        :param values: Variables values.
        :type values: List.
        :param dates: List of dates.
        :type dates: List.
        :return: Connection object or None
    """
    if conn is None:
        return None
    
    cur = conn.cursor()
    
    vals = []; 
    if values.ndim == 1:
        paths = 1
        n = 1
        k = 0
        vals.append(values.astype('str'))
    else:
        paths,n,k = values.shape
        for p in range(paths):
            v = values[p,:]
            vals.append(v.astype('str'))
    
    if not dates is None:
        dt = []
        ndates = len(dates)
        for i in range(ndates):
            dt.append('"' + str(dates[i]) + '"')
        n = min(n,ndates)
      
    try:
        for p in range(paths):
            for i in range(n):
                # Add single quotes around each
                if k == 0:
                    v = vals[i]
                else:
                    v = vals[p]
                    v = v[i,:]
                values_quotes = ["'" + value + "'" for value in v]
                values_txt = ", ".join(values_quotes)
                if dates is None:
                    query = "INSERT INTO DATA  VALUES (" + values_txt + ")"
                else:
                    query = "INSERT INTO DATA  VALUES (" + str(1+p) + "," + dt[i] + ", " + values_txt + ")"
                
                try:
                    cur.execute(query)
                except sqlite3.IntegrityError:
                    print('Could not insert: ')
                    print(values_txt)
                    sys.exit(-1)
    finally:    
        if conn: 
            conn.commit()
            conn.close()
            conn = None
        

def create_connection(dbfilename):
    """ 
    Create a database connection to the SQLite database specified by the dbfilename.
    
    Parameters:
        :param dbfilename: Path to db file.
        :type dbfilename: str.
        :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(dbfilename)
        return conn
    except Error as e:
        print(e,dbfilename)
        return None
   
    
def get_data(dbfilename, dt=None):
    """
    Execute query for a specific date or all dates if date is not specified
    
    Parameters:
        :param dbfilename: Path to db file.
        :type dbfilename: str.
        :param dt: Date time.
        :type dt: Datetime.
        :return: Results of query execution or None    
    """
    if dt is None:
        query = "SELECT * FROM DATA"
    else:
        query = "SELECT * FROM DATA WHERE date = '" + dt + "'"
    
    conn = create_connection(dbfilename)
    if conn is None:
        rows = None; columns = None
    else:
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(DATA)")
            columns = cur.fetchall()
            columns = [ col[1] for col in columns]
            cur.execute(query)
            rows = cur.fetchall()
        finally:
            conn.close()
            conn = None
    
    return rows,columns

def print_data(columns,data):
    """
    Return a pretty table that wraps data.
    
    Parameters:
        :param columns: Names of columns.
        :type columns: List.
        :param data: Data array.
        :type data: Datetime.
        :return: pretty table
    """
    
    from utils.prettyTable import PrettyTable
    pTable = PrettyTable(columns)
    for row in data:
        pTable.add_row(row)
        
    print(pTable)
 
    
def print_Data(columns,data):
    """
    Print data.
    
    Parameters:
        :param columns: Names of columns.
        :type columns: List.
        :param data: Data array.
        :type data: Datetime.
        :return:
    """
    cols = len(columns)
    rows = len(data)
    
    header = []
    for i in range(cols):
        h = columns[i]
        header.append(h[1]) 
        
    print(header)
    for i in range(rows):
        print(data[i])
       
# =============================================================================
#     row_format ="{:>15}" * (len(header) + 1)
#     print(row_format.format("", *header))
#     for row in data:
#         print(row_format.format(" ",*row))
# =============================================================================
           
      
def main(argv):
    """Main program."""
    print(argv)
    parser = argparse.ArgumentParser(description='Interface to sqlite database')
    parser.add_argument('-sqlite_dir', help="sqlite directory")
    parser.add_argument('-db', help="database name")
    args = parser.parse_args()
    sqlitedir = args.sqlite_dir
    dbfilename = args.db
    data,columns = get_data(sqlitedir, dbfilename)
    print_data(columns,data)
    return data
    
if __name__ == '__main__':
    """Main entry point."""
    main(sys.argv[1:])
