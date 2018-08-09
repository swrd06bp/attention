var express = require('express');
var app = express();
var server = require('http').Server(app);
var io = require('socket.io')(server);
var bodyParser = require('body-parser')

app.use(express.static(__dirname))

app.get('/', function (req, res) {
  res.sendFile(__dirname + '/index.html'); 
});

let image = '';
let boxes = '';
let prediction = '';
let logging = '';
let count_logging = 0;
let rewards = []
let accuracies = []

app.get('/reset', function (req, res){
  image = '';
  boxes = '';
  prediction = '';
  logging = '';
  count_logging = 0;
  rewards = []
  accuracies = []
});

app.post('/image', function (req, res){

  image = ''
  req.on('data', function (chunk) {
    image += chunk.toString();
  });
  req.on('end', function () {
    io.emit('image', unescape(image));
    res.sendStatus(200);
  });
});

app.post('/bboxes', function (req, res){
  boxes = ''
  req.on('data', function (chunk) {
    boxes += chunk.toString();
  });
  req.on('end', function () {
    io.emit('bboxes', JSON.parse(unescape(boxes)));
    res.sendStatus(200);
  });
});

app.post('/prediction', function (req, res){
  prediction = ''
  req.on('data', function (chunk) {
    prediction += chunk.toString();
  });
  req.on('end', function () {
    io.emit('prediction', JSON.parse(unescape(prediction)));
    res.sendStatus(200);
  });
});

app.post('/logging', function (req, res){
  req.on('data', function (chunk) {
    logging += chunk.toString();
  });
  req.on('end', function () {
    io.emit('logging', unescape(logging));
    count_logging += 1;
    if (count_logging > 50){
      logging = '';
      count_logging = 0;
    }
    res.sendStatus(200);
  });
});


app.post('/reward', function (req, res){
  reward = ''
  req.on('data', function (chunk) {
    reward += chunk.toString();
  });
  req.on('end', function () {
    rewards.push(JSON.parse(unescape(reward)));
    io.emit('reward', JSON.parse(unescape(reward)));
    res.sendStatus(200);
  });
});

app.post('/accuracy', function (req, res){
  accuracy = ''
  req.on('data', function (chunk) {
    accuracy += chunk.toString();
  });
  req.on('end', function () {
    accuracies.push(JSON.parse(unescape(accuracy)));
    io.emit('accuracy', JSON.parse(unescape(accuracy)));
    res.sendStatus(200);
  });
});

io.on('connection', function(socket){
  socket.emit('prediction', prediction);
  socket.emit('bboxes', boxes);
  socket.emit('image', image);
  socket.emit('logging', logging);
  socket.emit('rewards', rewards);
  socket.emit('accuracies', accuracies);
});

server.listen(8888, '0.0.0.0');

