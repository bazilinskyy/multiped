using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class canvasMover : MonoBehaviour
{

    public GameObject FollowingCanvas;
    public GameObject CameraTrget;
    public Vector3 cameraPos;
    public Vector3 canvas;
    public float distance = -2.0f;


    void Update()
    {
        cameraPos = CameraTrget.transform.position;
        canvas = FollowingCanvas.transform.position;

        FollowingCanvas.transform.position = new Vector3(cameraPos.x + distance, canvas.y, canvas.z);

    }
}
