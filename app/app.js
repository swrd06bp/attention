const express = require('express');
const app = express();
const server = require('http').Server(app);
const io = require('socket.io')(server);
const bodyParser = require('body-parser');
const publicIp = require('public-ip');

const args = process.argv;
const port = args[2];

app.use(express.static(__dirname))
app.use(bodyParser.urlencoded({ extended: false  }));
app.use(bodyParser.json())
app.set('view engine', 'ejs');

app.get('/', function (req, res) {
  publicIp.v4().then(ip => {
    console.log("Connection on http://" + ip + ":" + port)
    res.render(__dirname + '/index.ejs', {ip:ip, port:port}); 
  });
});

let image = '';
let boxes = '';
let prediction = '';
let logging = '';
let count_logging = 0;
let rewards = []
let accuracies = []

app.get('/reset', function (req, res) {
  image = '';
  boxes = '';
  prediction = '';
  logging = '';
  count_logging = 0;
  rewards = []
  accuracies = []
  res.sendStatus(200);
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
  io.emit('bboxes', req.body);
  res.sendStatus(200);
});

app.post('/prediction', function (req, res){
  io.emit('prediction', req.body);
  res.sendStatus(200);
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
  rewards.push(req.body);
  io.emit('reward', req.body);
  res.sendStatus(200);
});

app.post('/accuracy', function (req, res){
  accuracies.push(req.body);
  io.emit('accuracy', req.body);
  res.sendStatus(200);
});

io.on('connection', function(socket){
  socket.emit('prediction', prediction);
  socket.emit('bboxes', boxes);
  socket.emit('image', image);
  socket.emit('logging', logging);
  socket.emit('rewards', rewards);
  socket.emit('accuracies', accuracies);
});

server.listen(parseInt(port), '0.0.0.0');

