from flask import Flask, app
app =Flask(__name__)
@app.route('/')
def index():
    return '<h1>My Flask</h1>'

if __name__=='__main__':
    app.run(port=5000)
