There are **3 Servers running** here on PORTS 5000, 5001, 5002.

# PORT:5000
This server is responsible for QNN model
Located in file *app.py*. Other Contents of this are : Docker file, qnn_model.pkl, requirements.txt

# PORT:5001
This server is responsible for QVC model.
Located in *master.py*. Other Contents of this are : qvc_model.pkl, qvc_source_code.py

# PORT:5002
The **master**, that takes the input from API endpoint and passed then to 2 Servers (QVC, QNN) and aggregates the answer...
Located in *server.py* .

# How to Start ?
1. Clone the repo, navigate to project root dir
2. `cd qnn`
3. `docker build -t qnn-api .` build image
4. `docker run -p 5000:5000 qnn-api` run the service and map the PORT, don't close the terminal.
5. `cd ../qvc` navigate to QVC
6. `python3 -m venv venv`
7. `source venv/bin/activate`
8. `pip install flask numpy`
9. `python master.py` The service is running now, don't close this
10. `cd ensemble`
11. `python3 -m venv venv`
12. `source venv/bin/activate`
13. `pip install flask requests flask-cors`
14. `python server.py` The service is running now, don't close this.
15. Call the servive with `http://ipV4:5002/predict`, as *POST*, and body as 
```
payload = {
                "f1": data["f1"],
                "f2": data["f2"],
                "f3": data["f3"],
                "f4": data["f4"]
            }
```
and headers.