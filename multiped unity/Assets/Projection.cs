using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Projection : MonoBehaviour
{
    GameObject[] projections;
    public Vector3 endProjection;
    public GameObject zebra1;
    public GameObject zebra2;
    GameObject zebra;
    Transform projectionObject1;
    Transform projectionObject2;
    Transform projectionObject;
    public GameObject projection; 
    public GameObject line1; 
    public GameObject line2;
    GameObject line; 

    public Vector3[] Positions1;
    public Vector3[] Positions2;
    Vector3 endProjection1;
    Vector3 endProjection2;
    int counter;
    float frametime;

    CarMovement carMovementScript;

    // Start is called before the first frame update
    void Start()
    {
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        endProjection1 = Positions1[1];
        endProjection2 = Positions2[1];
        
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (carMovementScript.yielding)
        {
            counter += 1;
            if (counter == 1)
            {
                StartCoroutine(ProjectionLightRunner());
            }
            projectionObject.localPosition = Vector3.MoveTowards(projectionObject.transform.localPosition, endProjection, 10f * Time.fixedDeltaTime);
        }
    }
     
    
    IEnumerator ProjectionLightRunner()
    {
        frametime = 0.05f;
        projectionObject = projection.transform;
        

        if (carMovementScript.Yield == 1)
        {
            zebra = zebra1;
            line = line1; 
            projectionObject.transform.localPosition = Positions1[0];
            endProjection = endProjection1;
        }
        if (carMovementScript.Yield == 2)
        {
            line = line2; 
            zebra = zebra2;
            projectionObject.transform.localPosition = Positions2[0];
            endProjection = endProjection2;
        }
        projection.SetActive(true);
        
        
        if (counter == 1)
        {    
            yield return new WaitForSecondsRealtime(1f);
            line.SetActive(true);
            yield return new WaitForSecondsRealtime(3.6f);
            line.SetActive(false);
            projection.SetActive(false);
            yield return new WaitForSecondsRealtime(0.1f);
            zebra.SetActive(true);
            
            yield return new WaitForSecondsRealtime(2.7f);
            zebra.SetActive(false);
            yield return new WaitForSecondsRealtime(6.0f);
            counter = 0;
        }
        
    }
    

}
