var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);

app.use(express.static(__dirname))

app.get('/', function (req, res) {
  res.sendFile(__dirname + '/index.html'); 
});

app.post('/image', function (req, res){
  data = ''
  req.on('data', function (chunk) {
    data += chunk.toString();
  });
  req.on('end', function () {
    io.emit('image', unescape(data));
    res.sendStatus(200);
  });
});

app.post('/bboxes', function (req, res){
  data = ''
  req.on('data', function (chunk) {
    data += chunk.toString();
  });
  req.on('end', function () {
    io.emit('bboxes', unescape(data));
    res.sendStatus(200);
  });
});

server.listen(8888, '0.0.0.0');

