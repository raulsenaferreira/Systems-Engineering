// var Mongoose = require('mongoose');

// 	var Parse = require('parse').Parse;

// 	Parse.initialize("ww1iNl0YcOsoNwMMGdghQgouZ6vWVB2lCpOkWBZt", "0c3LgENpIqXaig3M2ApBX8eJ9lnp4n8yYgbxpREO");

// 	//banco
// 	var ReclamationDAO=Parse.Object.extend({className: "ReclamationDAO"});

// 	var query = new Parse.Query(ReclamationDAO);
// 	query.limit(1000);
// 	Mongoose.connect('mongodb://localhost/transreport', function (error) {
// 	    if (error) {
// 	        console.log(error);
// 	    }
// 	});
// 	  	var reportSchema = new Mongoose.Schema({
// 		  busNum: String,
// 		  busId: String,
// 		  busCom: String,
// 		  typeId: String,
// 		  type: String,
// 		  location: String,
// 		  date: String,
// 		  time: String,
// 		  image: String,
// 		  name: String,
// 		  email: String,
// 		  phone: String,
// 		  home: String,
// 		  createdAt: String
// 		});

// 		var ReportsModel = Mongoose.model('ReportsModel', reportSchema);
		
		
// 		//parse.com query	
		
		
// 		query.find({
// 			success: function(results){

// 				//fills objects with data from parse.com and save in MongoDB
// 				for (var i = 0; i < results.length; i++){

// 					var register = new ReportsModel({
// 				    busNum: results[i].get('bus_num'),
// 				    busId: results[i].get('bus_id'),
// 				    busCom: results[i].get('bus_com'),
// 				    typeId: results[i].get('type_id'),
// 				    type: results[i].get('type'),
// 				    location: results[i].get('location'),
// 				    date: results[i].get('date'),
// 				    time: results[i].get('time'),
// 				    image: results[i].get('image'),
// 				    name: results[i].get('name'),
// 				    email: results[i].get('email'),
// 				    phone: results[i].get('phone'),
// 				    home: results[i].get('home'),
// 				    createdAt: results[i].get('createdAt')
// 				  });

// 				  register.save(function(err, register) {
// 				    if (err) return console.error(err);
// 				    console.log(register);
// 				  });

// 				}
		
// 			},
// 			error: function(error) {
// 				console.log("Error: " + error.code + " " + error.message);
// 			}
// 		});


	
