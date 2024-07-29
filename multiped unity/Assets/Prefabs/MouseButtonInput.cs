using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class MouseButtonInput : MonoBehaviour
{

    public Slider Fill;
    private float FillSpeed = 5f;
    private float targetProgress = 0;
    public TextMeshProUGUI TopText;
    public TextMeshProUGUI BottomText;

    void Awake()
    {
        TopText.faceColor = new Color32(255, 255, 255, 60);
        BottomText.faceColor = new Color32(255, 255, 255, 255);
    }

    
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            targetProgress = 1;
            TopText.faceColor = new Color32(255, 255, 255, 255);
            BottomText.faceColor = new Color32(255, 255, 255, 20);
        }

        if (Input.GetMouseButtonUp(0))
        {
            targetProgress = 0;
            TopText.faceColor = new Color32(255, 255, 255, 20);
            BottomText.faceColor = new Color32(255, 255, 255, 255);
        }

        if (Fill.value < targetProgress)
            Fill.value += FillSpeed * Time.deltaTime;        

        if (Fill.value > targetProgress)
            Fill.value -= FillSpeed * Time.deltaTime;       
    }
}
