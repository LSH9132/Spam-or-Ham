const express = require("express");
const http = require("http");
const bodyParser = require("body-parser");
const request = require('request');
const path = require('path');;

const app = express();

app.use(bodyParser.urlencoded({ extended: false }));

const server = http.createServer(app);
const PORT = 8080;

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.urlencoded({extended: false}));

app.get('/', (req, res) => {
	res.sendFile(path.join(__dirname, 'public/index.html'));
});
app.post('/api', (req, res) => {
	// console.log(req.query);
	console.log(req.body);
	const content = req.body.content;
	try {
		request.post({
			url: 'http://aiserver:5000/',
			method: 'POST',
			json: {
				content: content,
			}
		}, (err, resp, body) => {
			const resd = resp.body
			console.log(resd);
			if(resd == 'True') {
				res.send('Spam');
				return
			}
			if(resd == 'False') {
				res.send('Ham');
				return
			}
			res.send('Error!');
			return
		});
	} catch (ex) {
		res.send('Error!');
	}
});

server.listen(PORT, () => {
	console.log(`Server running on http://localhost:${PORT}`);
});