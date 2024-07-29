using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Tracker : MonoBehaviour
{
    CarMovement carMovementScript;
    public Vector3[] Positions;
    int counter = 0;
    bool fading = false; 
    bool waitDone = false;
    Image image;
    Color tempColor; 

    // Start is called before the first frame update
    void Start()
    {
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        transform.localPosition = Positions[0];
        image = GetComponent<Image>();
        tempColor = image.color;
        tempColor.a = 0f;
        image.color = tempColor; 
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (carMovementScript.yielding)
        {       
            StartCoroutine(TrackerRunner());
            counter += 1; 
            transform.localPosition = Vector3.MoveTowards(transform.localPosition, Positions[1], 0.115f * Time.fixedDeltaTime);
        }
        if (fading)
        {                     
            transform.localPosition = Vector3.MoveTowards(transform.localPosition, Positions[0], 1.1f * Time.fixedDeltaTime);    
        }
    }

    IEnumerator TrackerRunner()
    {
        if(counter == 1)
        {
            tempColor.a = 1f;
            image.color = tempColor;
            yield return new WaitForSecondsRealtime(7.3f);

            fading = true;
            for (int i = 1; i < 30; i++)
            {
                tempColor.a = 1f - (i * 0.1f);
                image.color = tempColor;
                yield return new WaitForSecondsRealtime(0.003f);
            }
            yield return new WaitForSecondsRealtime(3f);
            fading = false; 
            transform.localPosition = Positions[0];
            counter = 0;
        }
    }
}
