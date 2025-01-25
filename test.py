import pyaudio

def list_device_info():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if "CABLE Output" in device_info["name"]:
            print(f"Device Info for {device_info['name']} (Index {i}):")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
            print(f"  Host API: {device_info['hostApi']}")
            print(f"  Supported Input Channels: {device_info['maxInputChannels']}")
    p.terminate()

list_device_info()