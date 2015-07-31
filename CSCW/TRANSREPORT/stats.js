var express = require('express');
var app = express();
var http = require('http').Server(app);
var bodyParser = require('body-parser');
var io = require('socket.io')(http);
var Mongoose = require('mongoose');

app.use("/css", express.static(__dirname + '/css'));
app.use("/js", express.static(__dirname + '/js'));
app.use("/img", express.static(__dirname + '/img'));
app.use("/fonts", express.static(__dirname + '/fonts'));
app.use("/font-awesome/css/", express.static(__dirname + '/font-awesome/css/'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));



app.post('/web',  function(req, res){

  var register = new ReportsModel({
    busNum: req.body.bus_num,
    busId: req.body.bus_id,
    busCom: req.body.bus_com,
    typeId: req.body.type_id,
    type: req.body.type,
    location: req.body.location,
    date: req.body.date,
    time: req.body.time,
    image: req.body.image,
    name: req.body.name,
    email: req.body.email,
    phone: req.body.phone,
    home: req.body.home,
    createdAt: req.body.created_at
  });

  register.save(function(err, register) {
    if (err) return console.error(err);
    res.send(register);
  });

});

Mongoose.connect('mongodb://localhost/transreport', function (error) {
    if (error) {
        console.log(error);
    }
});

var statsSchema = new Mongoose.Schema({
  busNum: String,
  idOcc: Number,
  occType: String,
  qty: Number
});

var reportSchema = new Mongoose.Schema({
  busNum: String,
  busId: String,
  busCom: String,
  typeId: String,
  type: String,
  location: String,
  date: String,
  time: String,
  image: String,
  name: String,
  email: String,
  phone: String,
  home: String,
  createdAt: String
});

var Statistic = Mongoose.model('Statistic', statsSchema);
var ReportsModel = Mongoose.model('ReportsModel', reportSchema);

app.get('/registros/', function(req,res){
    
    ReportsModel.find({},function(err, reports) {
      if (err) return console.error(err);
      //var html = '<br><br><br><br><h3>Feed Notícias</h3>';
      var html='<br>';
      for (i in reports){
        if (reports[i].busNum) {
          //html+='---------------------------------------------<br>';
          html+= '<div class="report content-section-a col-lg-5 col-md-5 col-sm-3 col-md-offset-1">';
          if (reports[i].email) {
            html+='<p class="image">Usuário: '+reports[i].email+'</p>';
          }
          html+='<p class="line">Linha '+reports[i].busNum;
          if (reports[i].busCom) {
            html+=' - '+reports[i].busCom;
          }
          if (reports[i].busId) {
            html+=' - '+reports[i].busId;
          }
          if (reports[i].type) {
            html+='</p><p class="type">'+reports[i].type+'</p>';
          }
          if (reports[i].location) {
            html+='<p class="location">'+reports[i].location+'</p>';
          }
          if (reports[i].date) {
            html+='<p class="date">'+reports[i].date+'</p>';
          }
          if (reports[i].time) {
            html+='<p class="time">'+reports[i].time+'</p>';
          }
          if (reports[i].image) {
            html+='<p class="image"><img src="'+reports[i].image+'"></p>';
          }
          html+='</div>';
          //html+='---------------------------------------------';
        }
                
      }
      res.send(html); 
    });
});

app.get('/', function (req, res) {
  res.sendfile('index.html');
});

app.get('/stats/', function (req, res) {
  res.sendfile('stats.html');
});

app.get('/news/', function (req, res) {
  res.sendfile('news.html');
});

app.get('/stats/types/', function (req, res) {
  
  ReportsModel.find({}, function(err, stats) {
    mapTypes = {}
    arrayTypes=''
    if (err) return console.error(err);
    
    for (i in stats){
      mapTypes[stats[i].typeId]=stats[i].type;       
    }

    for(m in mapTypes){
      if(m != "undefined")
        arrayTypes+='<li><a class="typeOcc" value='+m+'>'+mapTypes[m]+'</a></li>'
    }

    res.send(arrayTypes);
    
  });
});

app.get('/total/', function(req,res){

    ReportsModel.find({},function(err, stats) {
      if (err) return console.error(err);      
      res.send(JSON.stringify(calculateTopBusline(stats, 20)));
    });
});

app.get('/totalByOcc/', function(req,res){

    ReportsModel.find({},function(err, stats) {
      if (err) return console.error(err);      
      res.send(JSON.stringify(calculateTopOccurrences(stats, 20)));
    });
});

app.get('/tipo/:tipo', function(req,res){
    
    ReportsModel.find({typeId: req.params.tipo},function(err, stats) {
      if (err) return console.error(err);
      res.send(JSON.stringify(calculateTopBusline(stats, 20)));
    });
});

function calculateTopOccurrences(stats, top){
  arrayType=[];
  maxOccurrences={};
  
  for (var i = 0; i < stats.length; i++) {
    var key = stats[i].type;
    if (key) {key=key.toLowerCase();}
    
    if(maxOccurrences[key]==undefined) maxOccurrences[key]=1; 
    else maxOccurrences[key]++;
  };
      
  var mapKeys = Object.keys(maxOccurrences);
  mapKeys.sort(function(a,b){
    return maxOccurrences[b] - maxOccurrences[a];
  });

  var cont=0;
  mapKeys.forEach(function(k){
    if(cont < top && k != "undefined"){
      arrayType.push({ label: k+' ('+maxOccurrences[k]+' casos)',
       instances: maxOccurrences[k] });
    }
    cont++;
  });

  return arrayType;
}

function calculateTopBusline(stats, top){
  arrayBus=[];
  maxOccurrences={};
  
  for (var i = 0; i < stats.length; i++) {
    var key = stats[i].busNum;
    if(maxOccurrences[key]==undefined) maxOccurrences[key]=1; 
    else maxOccurrences[key]++;
  };
      
  var mapKeys = Object.keys(maxOccurrences);
  mapKeys.sort(function(a,b){
    return maxOccurrences[b] - maxOccurrences[a];
  });

  var cont=0;
  mapKeys.forEach(function(k){
    if(cont < top && k != "undefined"){
      arrayBus.push({ busNum: k, qty: maxOccurrences[k] });
    }
    cont++;
  });

  return arrayBus;
}

http.listen(4000, function(){
  console.log('Disponível na porta :4000');
});
