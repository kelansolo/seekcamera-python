#!/usr/bin/env python3
# Copyright 2021 Seek Thermal Inc.
#
# Original author: Michael S. Mead <mmead@thermal.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import sleep

import numpy as np
import datetime as dt

from seekcamera import (
    SeekCameraIOType,
    SeekCameraManager,
    SeekCameraManagerEvent,
    SeekCameraFrameFormat,
    SeekFrame
)
start = -1
stop = -1
timestamps = []

def on_frame(camera, camera_frame, file):
    """Async callback fired whenever a new frame is available.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera for which the new frame is available.
    camera_frame: SeekCameraFrame
        Reference to the class encapsulating the new frame (potentially
        in multiple formats).
    file: TextIOWrapper
        User defined data passed to the callback. This can be anything
        but in this case it is a reference to the open CSV file to which
        to log data.
    """
    global timestamps
    frame = camera_frame.thermography_float
    
    timestamps.append(frame.header.timestamp_utc_ns)

    print(
        "frame available: {cid} (size: {w}x{h})".format(
            cid=camera.chipid, w=frame.width, h=frame.height
        )
    )

    # Append the frame to the CSV file.
    np.savetxt(file, frame.data, fmt="%.1f")

def on_event(camera, event_type, event_status, _user_data):
    """Async callback fired whenever a camera event occurs.

    Parameters
    ----------
    camera: SeekCamera
        Reference to the camera on which an event occurred.
    event_type: SeekCameraManagerEvent
        Enumerated type indicating the type of event that occurred.
    event_status: Optional[SeekCameraError]
        Optional exception type. It will be a non-None derived instance of
        SeekCameraError if the event_type is SeekCameraManagerEvent.ERROR.
    _user_data: None
        User defined data passed to the callback. This can be anything
        but in this case it is None.
    """
    global stop
    global start
    print("{}: {}".format(str(event_type), camera.chipid))

    if event_type == SeekCameraManagerEvent.CONNECT:
        # Open a new CSV file with the unique camera chip ID embedded.
        try:
            file = open("thermography-" + camera.chipid + ".csv", "w")
        except OSError as e:
            print("Failed to open file: %s" % str(e))
            return

        # Start streaming data and provide a custom callback to be called
        # every time a new frame is received.
        camera.register_frame_available_callback(on_frame, file)
        start = dt.datetime.now()

        camera.capture_session_start(SeekCameraFrameFormat.THERMOGRAPHY_FLOAT)


    elif event_type == SeekCameraManagerEvent.DISCONNECT:       
        # stop = dt.datetime.now()
        camera.capture_session_stop()

    elif event_type == SeekCameraManagerEvent.ERROR:
        print("{}: {}".format(str(event_status), camera.chipid))

    elif event_type == SeekCameraManagerEvent.READY_TO_PAIR:
        return 

def main():
    # Create a context structure responsible for managing all connected USB cameras.
    # Cameras with other IO types can be managed by using a bitwise or of the
    # SeekCameraIOType enum cases.
    runtime = 3
    global start
    global stop
    with SeekCameraManager(SeekCameraIOType.USB) as manager:
        # Start listening for events.
        manager.register_event_callback(on_event)
        while dt.datetime.now() - start <= dt.timedelta(seconds=5):
            sleep(0.1)

if __name__ == "__main__":
    main()

def get_data(runtime):
    # Create a context structure responsible for managing all connected USB cameras.
    # Cameras with other IO types can be managed by using a bitwise or of the
    # SeekCameraIOType enum cases.
    global start
    global stop
    global timestamps
    start = -1
    stop = -1
    timestamps = []
    with SeekCameraManager(SeekCameraIOType.USB) as manager:
        # Start listening for events.
        manager.register_event_callback(on_event)
        if True:
            sleep(runtime)
        stop = dt.datetime.now()
    return(timestamps)