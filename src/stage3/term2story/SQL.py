# coding: utf-8
## This should be run on camel
import pymysql
db = pymysql.connect("localhost","root","iloveyou","VIST" )
cursor = db.cursor()
def find_path(sub, obj):
    cursor.execute(f"select * from Visual_Genome_relationship\
                        where Subject=\"{sub}\" and Object=\"{obj}\"")
    return cursor.fetchall()

