using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class canvasMover : MonoBehaviour
{

    public GameObject FollowingCanvas;
    public GameObject CameraTrget;
    public Vector3 cameraPos;
    public Vector3 canvas;
    public Vector3 distance = new Vector3(-2.0f, 0f, 0f);
    public Vector3 rotationOffset = new Vector3(0f, 0f, 0f);


    void Update()
    {
        cameraPos = CameraTrget.transform.position;
        canvas = FollowingCanvas.transform.position;

        FollowingCanvas.transform.position = new Vector3(
            cameraPos.x + distance.x,
            cameraPos.y + distance.y, 
            cameraPos.z + distance.z
            );

        FollowingCanvas.transform.rotation = Quaternion.Euler(
            CameraTrget.transform.rotation.eulerAngles.x + rotationOffset.x,
            CameraTrget.transform.rotation.eulerAngles.y + rotationOffset.y,
            CameraTrget.transform.rotation.eulerAngles.z + rotationOffset.z
            );

    }
}
