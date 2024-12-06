using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ProgressBar : MonoBehaviour
{
    CarMovement carMovementScript;
    public GameObject progressAnimation;
    int counter; 

    void Start()
    {
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();
        progressAnimation.SetActive(false); 
    }

    void FixedUpdate()
    {
        if(carMovementScript.yielding)
        {
            counter += 1;
            if (counter == 1)
            {
                StartCoroutine(Wait());
            }     
        }      
    }

    IEnumerator Wait()
    {
        progressAnimation.SetActive(true);
        yield return new WaitForSecondsRealtime(7.6f);
        progressAnimation.SetActive(false);
        counter = 0; 
    }

}

