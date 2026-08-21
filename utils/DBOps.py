from typing import Literal
import sqlalchemy
import pandas
import os

connectionString = "DRIVER=ODBC Driver 17 for SQL Server;SERVER=AiRTS-SQLSERVER;DATABASE=NUSPhase2a;UID=1fssuser;PWD=Q12024qeadzc;"

engine = sqlalchemy.create_engine("mssql+pyodbc://?odbc_connect=%s" % connectionString)

def ReadSQL(tableName: str, row: str = "") -> pandas.DataFrame:
    try:
        with engine.connect() as connection:
            query = f"SELECT {row} * FROM {tableName}"
            return pandas.read_sql(query, connection)
        
    except Exception as e:
        print("Error:", e)
    return pandas.DataFrame()

def InsertTable(DF: pandas.DataFrame, tableName: str, if_exist: Literal["replace", "append"] = "replace") -> None:
    try: 
        print(DF.head(5))
        DF.to_sql(tableName, engine, if_exists=if_exist, index=False)
        print(f"Successfully inserted {tableName}")
    
    except Exception as e:
     print("Error:", e)

def DropStagingTables() -> None:
    try:
        databaseInspector = sqlalchemy.inspect(engine)

        tableNames = databaseInspector.get_table_names()
        tableMeta = sqlalchemy.MetaData()

        for name in tableNames:
            if 'raw' not in name.lower():
                table = sqlalchemy.Table(name, tableMeta, autoload_with=engine)

                table.drop(engine)
                print(f"Table dropped: {name}")

    except Exception as e:
        print("Error:", e)

def SQLToCSV() -> None:
    try:
        with engine.connect() as conn:
            databaseInspector = sqlalchemy.inspect(engine)
            pass
            tableNames = databaseInspector.get_table_names()

            for table in tableNames:
                if "STGOUT_" in table or "OUT_" in table:
                    DF = pandas.read_sql_table(table, engine)
                    DF.to_csv("./data/exported/" + table + ".csv")

    except Exception as e:
        print("Error:", e)

def VendorToSQL() -> None:
    headers = [
        "Vendor", 
        "Cty", 
        "Name 1", 
        "Name 2", 
        "Name 3", 
        "Name 4", 
        "City", 
        "District", 
        "PO Box", 
        "PO Box pcd", 
        "PostalCode", 
        "Rg", 
        "Street", 
        "Address", 
        "Group.1", 
        "Telephone 1",
        "Telephone 2",
        "Fax Number",
        "VAT Registration No.",
        "URL",
        "CoCd",
        "Bank Account",
        "Created by"
    ]
    toImportPath = "./data/ToImport"
    try:
        compiledDF: pandas.DataFrame = pandas.DataFrame()
        for fileName in os.listdir(toImportPath):
            DF = pandas.read_excel(os.path.join(toImportPath, fileName))
            print(f"Read {fileName}")
            DF = DF[headers]
            DF.rename(columns={"Group.1": "Group"}, inplace=True)
            if compiledDF.empty:
                print(f"Assigning {fileName} as first DataFrame")
                compiledDF = DF
                print(compiledDF.size)
            else:
                print(f"Appending {fileName} to DataFrame")
                compiledDF = pandas.concat([compiledDF, DF], ignore_index=True)
                print(compiledDF.size)

        tableName = "Z001RawVendorMaster"
        print(DF.head(20))
        print(f"Inserting into {tableName}")

        compiledDF.to_sql(tableName, engine, if_exists="replace", index=False)
        print(f"Successfully inserted {tableName}")

    except Exception as e:
        print("Error:", e)

def CSVToSQL():
    toImportPath = "./data/ToImport"
    try:
        for fileName in os.listdir(toImportPath):
            DF = pandas.read_csv(os.path.join(toImportPath, fileName))
            print(f"Read {fileName}")
            tableName = fileName.split(".")[0]
            print(f"Inserting into {tableName}")
            DF.to_sql(tableName, engine, if_exists="replace", index=False)
            print(f"Successfully inserted {tableName}")

    except Exception as e:
        print("Error:", e)

def GetNonDuplicates(table: str, duplicateTable: str, feature: str = '') -> pandas.DataFrame:
    try:
        additionalQuery = f"AND [{feature}] != ''"
        with engine.connect() as connection:
            query = f"SELECT DISTINCT * FROM [{table}] WHERE [Vendor] NOT IN (SELECT [Vendor] FROM [{duplicateTable}]) {additionalQuery}"
            return pandas.read_sql(query, connection)
        
    except Exception as e:
        print("Error:", e)
    return pandas.DataFrame()





