path=dirname(Base.source_path());
movies = readdlm("$path/data/ml-100k/u.item", '|');
prefs = readdlm("$path/data/ml-100k/u.data", '\t');
users = readdlm("$path/data/ml-100k/u.user", '|');
#print (movies[1]);
#print (prefs[1]);
#print (users[1]);

#using ODBC
#connection = advancedconnect("Driver={psqlODBC};user=raul;server=localhost;database=datamining;")
