using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Pixelplacement;

public class LightStripBumper : MonoBehaviour
{
    // Animation
    //float StartAni = 3.95f;
    float StartAni = 0f;
    float NumAni = 8;
    float opacity = 1f;
    float bandTime = 1.2f;

    Gradient myLine;
    LineRenderer lineRenderer;
    public Transform[] Anchors;
    public Color myColor;
    CarMovement carMovementScript; 
    // To check if block has started
    public bool bumperStarted = false;
    private bool bumperAnimation = false;
    public int counter = 0;
    public int counter2 = 0;
    ConditionController conditionScript; 
    // Use this for initialization
    void Start()
    {
        carMovementScript = GameObject.Find("CarMovement").GetComponent<CarMovement>();  
        conditionScript = GameObject.Find("ConditionController").GetComponent<ConditionController>();
    }
      
    // Update is called once per frame
    void FixedUpdate()
    {
        if (conditionScript.eHMIOn == 1)
        {
            // Start HMI when block is started
            if (bumperStarted)
            {
                lineRenderer.colorGradient = myLine;
            }
            if (carMovementScript.WaveStarted)
            {
                counter += 1;
                bumperStarted = true;
            }
            if (counter == 1)
            {
                HMI();
                counter += 1;
            }

            if (carMovementScript.yielding)
            {
                counter2 += 1;
            }
            if (counter2 == 1)
            {
                yieldHMI();
                counter2 += 1;
            }
        }
    }

    // HMI
    public void HMI()
    {
        //draw linerender
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = Anchors.Length;
        var t = Time.time;
        for (int i = 0; i < Anchors.Length; i++)
        {
            lineRenderer.SetPosition(i, Anchors[i].position);
        }
        lineRenderer.Simplify(0);
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        myLine = CreateFullBand();
        
    } 

    // Yielding HMI
    public void yieldHMI()
    {  
            //fullband disappear
            LeanTween.value(1.0f, 0.0f, 0f)
                         .setDelay(StartAni)
                         .setEase(LeanTweenType.easeInOutSine)
                                .setOnUpdate((float val) =>
                                {
                                    opacity = val;
                                    myLine = CreateFullBand();
                                });

            //yielding animation
            for (int i = 0; i < NumAni; i++)
            {
                LeanTween.value(0.0f, 0.95f, bandTime)       // start position, end position, time to cover. 0.0 = outside, 1.0 = inside (middle)
                         .setDelay(StartAni + bandTime * i * 0.8f)
                         .setEase(LeanTweenType.linear)
                               .setOnUpdate((float val) =>
                               {
                                   opacity = 1;
                                   myLine = CreateGradient(0.1f, val);  // length of the light strip
                               });
            }

            //fullband reappear
            LeanTween.value(0.0f, 1.0f, 0.5f)
                     .setDelay(StartAni + bandTime * NumAni * 0.8f)
                     .setEase(LeanTweenType.easeInOutSine)
                            .setOnUpdate((float val) =>
                            {
                                opacity = 1;
                                myLine = CreateFullBand();
                            });
        
    }
    
    // Non-Yielding HMI
    public void driveHMI()
    {
        //draw linerender
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = Anchors.Length;
        //var t = Time.time;
        for(int i = 0; i < Anchors.Length; i++) {
            lineRenderer.SetPosition(i, Anchors[i].position);
        }

        lineRenderer.Simplify(0);
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        myLine = CreateFullBand();
    }

    //for yielding animation
    Gradient CreateGradient(float length, float offset)
    {

        float colOut = Mathf.Clamp(0f - length + offset, 0, 0.5f);
        float colIn = Mathf.Clamp(length - length + offset, 0, 0.5f);

        Gradient gradient = new Gradient();

        gradient.SetKeys(
            new GradientColorKey[] {
            new GradientColorKey(myColor, colOut), new GradientColorKey(myColor, colIn),
            new GradientColorKey(myColor, 1f - colIn), new GradientColorKey(myColor, 1f - colOut)
        },

            new GradientAlphaKey[] {
            //left line
            new GradientAlphaKey(0.0f, colOut),
            new GradientAlphaKey(opacity, colOut), new GradientAlphaKey(opacity, colIn),
            new GradientAlphaKey(0.0f, colIn),
            //right line
            new GradientAlphaKey(0.0f, 1f- colIn),
            new GradientAlphaKey(opacity, 1f- colIn), new GradientAlphaKey(opacity, 1f- colOut),
            new GradientAlphaKey(0.0f, 1.0005f- colOut)
        });

        return gradient;
    }

    //for fullband
    Gradient CreateFullBand()
    {
        Gradient gradient = new Gradient();

        gradient.SetKeys(
            new GradientColorKey[] {
            new GradientColorKey(myColor, 0.0f), new GradientColorKey(myColor, 1.0f)
        },

            new GradientAlphaKey[] {
            new GradientAlphaKey(opacity, 0.0f), new GradientAlphaKey(opacity, 1.0f)

        });

        return gradient;
    }
}