# # # [Library for MQTT]
from PIL import Image
from io import BytesIO
from base64 import b64encode
from paho.mqtt import client as mqtt_client
import json
import time
import datetime
import pytz
import cv2


def connect_mqtt(broker, port, client_id):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to ", broker, " port: ", port)
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def mqttInit(mqttPipe, endQueue):
    
    # Set parameters for mqtt client
    broker = 'broker.emqx.io'
    port = 1883
    topic = 'hmh-pga-internship-2020'
    client_id = '111199'

    # Initialize client
    client = connect_mqtt(broker, port, client_id)
    # client.loop_start()
    
    while True:
        collection = mqttPipe.recv()
        if collection is None:
            break
        [frame, line, place, centroid] = collection

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(frame, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 255, 255), 2)
        evidence = Image.fromarray(frame, 'RGB')
        buffer = BytesIO()
        evidence.save(buffer, format = "JPEG")
        base64Image = str(b64encode(buffer.getvalue()))

        start = time.time()
        now = datetime.datetime.utcnow()
        now = now.replace(tzinfo = pytz.UTC)
        message = json.dumps({"usr": "hoanghm2","cam_id": "SV1", "line": line, "direction": place ,"time": now.isoformat(), "evidence": base64Image})
        result = client.publish(topic, message)


    print("MQTT DONE")