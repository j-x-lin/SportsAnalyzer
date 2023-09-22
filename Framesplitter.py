import cv2


def video_to_frames(video_path, count):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cv2.imwrite('data/frames/%d.jpg' % count, image)
        success, image = vidcap.read()
        count += 1

    return count


count = 0

f = open("data/index", "r")
while True:
    line = f.readline()

    if not line:
        break

    count = video_to_frames('data/videos/' + line + '.mp4', count)
    print('Done with', line)
