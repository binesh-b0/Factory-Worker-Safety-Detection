import numpy as np
import cv2
from classify import get_prep_img, predict_safety

if __name__ == "__main__":
    classes = None
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    Width = 640
    Height = 480

    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    cv2.startWindowThread()

    # open webcam video stream
    cap = cv2.VideoCapture(0)

    # the output will be written to output.avi
    out_vid = cv2.VideoWriter(
        'output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))

        net.setInput(cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

        for i in indices:
            i = i
            box = boxes[i]
            if class_ids[i] == 0:
                label = str(classes[class_id])
                cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(box[0] + box[2]), round(box[1] + box[3])),
                              (0, 0, 0), 2)
                prepped_img = get_prep_img(frame)
                predicted_val = predict_safety(prepped_img)
                cv2.putText(frame, predicted_val, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 2)

        # Write the output video
        out_vid.write(frame.astype('uint8'))
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    # and release the output
    out_vid.release()
    # finally, close the window
    cv2.destroyAllWindows()
    cv2.waitKey(1)