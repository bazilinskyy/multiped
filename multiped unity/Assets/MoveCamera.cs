using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveCamera : MonoBehaviour
{

    public Vector3[] Positions;
    public float Speed;
    public bool StartWalk = false;
 
    void Update()
    {
        if (StartWalk)
        {
            Vector3 endPos = Positions[1];
            transform.position = Vector3.MoveTowards(transform.position, endPos, Speed * Time.deltaTime);
        }       
    }
}
