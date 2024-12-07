using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class SliderValueScript : MonoBehaviour
{
    public TMP_Text text;
    public Slider slider;

    private void Awake()
    {
        slider = GetComponentInParent<Slider>();
        
    }

    private void Start()
    {
        UpdateText(slider.value);
        slider.onValueChanged.AddListener(UpdateText);
    }
    void UpdateText(float val)
    {
        text.text = val.ToString();
    }
}

