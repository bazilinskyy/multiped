using UnityEngine;
using System.Collections;
using UnityEngine.UI;

public class CountDown : MonoBehaviour
{
    Image fillImg;
    float timeAmt = 3;
    float time;
    float timeDisplay;
    public Text timeText;
    public bool timer; 
    public bool timerCompleted = false; 

    // Use this for initialization
    void OnEnable()
    { 
        fillImg = this.GetComponent<Image>();
        time = timeAmt;
        timer = true;
        timerCompleted = false; 
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        if (timer)
        {
            if (time > 0)
            {
                time -= Time.fixedDeltaTime;

                fillImg.fillAmount = time / timeAmt;
                timeDisplay = Mathf.Ceil(time);
                timeText.text = timeDisplay.ToString("F0");
            }
            if (time < 0.01)
            {
                timeText.text = "GO!";
                timer = false;
                timerCompleted = true; 
            }
        }
        if (timerCompleted)
        {
            StartCoroutine(Wait()); 
           
           
        }
    }
    IEnumerator Wait()
    {
       yield return new WaitForSecondsRealtime(1);
        GameObject.Find("CountDown").SetActive(false);
    }
}