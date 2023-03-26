from spam import predict_spam

from flask import Flask, request

app = Flask(__name__)

@app.route('/', methods=['POST'])
def root():
  data = request.json
  spamdata = predict_spam(data.get('content'))
  if spamdata > 0.7:
    return 'True', 200
  else:
    return 'False', 200
  # return 'Error!', 500

if __name__ == "__main__":
  app.run(host="0.0.0.0", port="5000")