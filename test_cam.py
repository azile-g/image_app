import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from webcam import webcam
import av
import threading
import cv2

#def callback(frame):
#    img = frame.to_ndarray(format="bgr24")
#    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
#    return av.VideoFrame.from_ndarray(img, format="bgr24")
#webrtc_streamer(key="example", video_frame_callback=callback)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Streamlit WebRTC Demo", page_icon="🤖")
task_list = ["Video Stream", "Video Capture", "File Upload"]

with st.sidebar:
    st.title("")
    task_name = st.selectbox("Select your tasks:", task_list)
st.title(task_name)

if task_name == task_list[1]:
    captured_image = webcam()
    if captured_image is None:
        st.write("Waiting for capture...")
    else:
        st.write("Got an image from the webcam:")
        st.image(captured_image)

if task_name == task_list[2]: 
    st.file_uploader("Upload Your Photos Here:")

if task_name == task_list[0]:
    style_list = ['color', 'black and white']
    st.sidebar.header('Style Selection')
    style_selection = st.sidebar.selectbox("Choose your style:", style_list)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model_lock = threading.Lock()
            self.style = style_list[0]

        def update_style(self, new_style):
            if self.style != new_style:
                with self.model_lock:
                    self.style = new_style

        def recv(self, frame):
            img = frame.to_image()
            if self.style == style_list[1]:
                img = img.convert("L")
            return av.VideoFrame.from_image(img)

    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )

    if ctx.video_processor:
        ctx.video_transformer.update_style(style_selection)

