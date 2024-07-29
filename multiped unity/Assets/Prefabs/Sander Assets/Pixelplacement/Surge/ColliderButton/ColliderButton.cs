/// <summary>
/// SURGE FRAMEWORK
/// Author: Bob Berkebile
/// Email: bobb@pixelplacement.com
/// 
/// Simple system for turning anything into a button.
/// 
/// </summary>

using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;
using UnityEngine.EventSystems;
using System;
using Pixelplacement.TweenSystem;
#if UNITY_2017_2_OR_NEWER
using UnityEngine.XR;
#else
using UnityEngine.VR;
#endif

namespace Pixelplacement
{
	[RequireComponent(typeof(Collider))]
	[RequireComponent(typeof(Rigidbody))]
	[ExecuteInEditMode]
	public class ColliderButton : MonoBehaviour
	{
		#region Public Events
		public GameObjectEvent OnSelected;
		public GameObjectEvent OnDeselected;
		public GameObjectEvent OnClick;
		public GameObjectEvent OnPressed;
		public GameObjectEvent OnReleased;
		#endregion

		#region Public Enums
		public enum EaseType { EaseOut, EaseOutBack };
		#endregion

		#region Public Properties
		public bool IsSelected
		{
			get;
			private set;
		}
		#endregion

		#region Public Variables
		public KeyCode[] keyInput;
		public bool _unityEventsFolded;
		public bool _scaleResponseFolded;
		public bool _colorResponseFolded;
		public bool applyColor = true;
		public bool applyScale;
		public LayerMask collisionLayerMask = -1;
		public Renderer colorRendererTarget;
		public Image colorImageTarget;
		public Color normalColor = Color.white;
		public Color selectedColor = Color.gray;
		public Color pressedColor = Color.green;
		public float colorDuration = .1f;
		public Transform scaleTarget;
		public Vector3 normalScale;
		public Vector3 selectedScale;
		public Vector3 pressedScale;
		public float scaleDuration = .1f;
		public EaseType scaleEaseType;
		public bool resizeGUIBoxCollider = true;
		public Vector2 guiBoxColliderPadding;
		#endregion

		#region Private Variables
		bool _clicking;
		int _selectedCount;
		bool _colliderSelected;
		bool _pressed;
		bool _released;
		bool _vrRunning;
		RectTransform _rectTransform;
		EventTrigger _eventTrigger;
		EventTrigger.Entry _pressedEventTrigger;
		EventTrigger.Entry _releasedEventTrigger;
		EventTrigger.Entry _enterEventTrigger;
		EventTrigger.Entry _exitEventTrigger;
		int _colliderCount;
		BoxCollider _boxCollider;
		TweenBase _colorTween;
		TweenBase _scaleTween;
		#endregion

		#region Init
		private void Reset()
		{
			keyInput = new KeyCode[] { KeyCode.Mouse0 };

			//color setup:
			colorRendererTarget = GetComponent<Renderer>();
			colorImageTarget = GetComponent<Image>();

			//scale setup:
			scaleTarget = transform;
			normalScale = transform.localScale;
			selectedScale = transform.localScale * 1.15f;
			pressedScale = transform.localScale * 1.25f;

			//set initial size on gui collider:
			_rectTransform = GetComponent<RectTransform>();
			_boxCollider = GetComponent<BoxCollider>();
			if (_rectTransform != null && _boxCollider != null) ResizeGUIBoxCollider(_boxCollider);

			//set up rigidbody:
			GetComponent<Rigidbody>().isKinematic = true;
		}

		private void Awake()
		{
			if (!Application.isPlaying) return;

			//rect and event triggers:
			_rectTransform = GetComponent<RectTransform>();
			if (_rectTransform != null) 
			{
				_eventTrigger = gameObject.AddComponent<EventTrigger>();
				_pressedEventTrigger = new EventTrigger.Entry();
				_pressedEventTrigger.eventID = EventTriggerType.PointerDown;
				_releasedEventTrigger = new EventTrigger.Entry();
				_releasedEventTrigger.eventID = EventTriggerType.PointerUp;
				_enterEventTrigger = new EventTrigger.Entry();
				_enterEventTrigger.eventID = EventTriggerType.PointerEnter;
				_exitEventTrigger = new EventTrigger.Entry();
				_exitEventTrigger.eventID = EventTriggerType.PointerExit;
			}

			//collider cache:
			_boxCollider = GetComponent<BoxCollider>();
		}
		#endregion

		#region Flow
		private void OnEnable()
		{
			if (!Application.isPlaying) return;
		
			if (_rectTransform != null)
			{
				//event registrations:
				_pressedEventTrigger.callback.AddListener((data) => { OnPointerDownDelegate((PointerEventData)data); });
				_eventTrigger.triggers.Add(_pressedEventTrigger);
				_releasedEventTrigger.callback.AddListener((data) => { OnPointerUpDelegate((PointerEventData)data); });
				_eventTrigger.triggers.Add(_releasedEventTrigger);
				_enterEventTrigger.callback.AddListener((data) => { OnPointerEnterDelegate((PointerEventData)data); });
				_eventTrigger.triggers.Add(_enterEventTrigger);
				_exitEventTrigger.callback.AddListener((data) => { OnPointerExitDelegate((PointerEventData)data); });
				_eventTrigger.triggers.Add(_exitEventTrigger);
			}
		}

		private void OnDisable()
		{
			if (!Application.isPlaying) return;

			//resets:
			_pressed = false;
			_released = false;
			_clicking = false;
			_colliderSelected = false;
			_selectedCount = 0;
			_colliderCount = 0;

			ColorReset();
			ScaleReset();

			if (_rectTransform != null)
			{
				//event deregistrations:
				_pressedEventTrigger.callback.RemoveAllListeners();
				_eventTrigger.triggers.Remove(_pressedEventTrigger);
				_releasedEventTrigger.callback.RemoveAllListeners();
				_eventTrigger.triggers.Remove(_releasedEventTrigger);
				_enterEventTrigger.callback.RemoveAllListeners();
				_eventTrigger.triggers.Remove(_enterEventTrigger);
				_exitEventTrigger.callback.RemoveAllListeners();
				_eventTrigger.triggers.Remove(_exitEventTrigger);
			}
		}
		#endregion

		#region Loops
		private void Update()
		{
			//update gui colliders:
			if (resizeGUIBoxCollider && _rectTransform != null && _boxCollider != null)
			{
				//fit a box collider:
				if (_boxCollider != null)
				{
					ResizeGUIBoxCollider(_boxCollider);
				}
			}

			//for in editor updating of the gui collider:
			if (!Application.isPlaying) return;

			//VR status:
#if UNITY_2017_2_OR_NEWER
			_vrRunning = (XRSettings.isDeviceActive);
#else
			_vrRunning = (VRSettings.isDeviceActive);
#endif
		
			//collider collision started:
			if (!_colliderSelected && _colliderCount > 0)
			{
				_colliderSelected = true;
				Selected();
			}

			//collider collision ended:
			if (_colliderSelected && _colliderCount == 0)
			{
				_colliderSelected = false;
				Deselected();
			}

			//process input:
			if (keyInput != null && _selectedCount > 0)
			{
				foreach (var item in keyInput)
				{
					if (Input.GetKeyDown(item))
					{
						if (_selectedCount == 0) return;
						Pressed();
					}

					if (Input.GetKeyUp(item))
					{
						Released();
					}
				}
			}
		}
		#endregion

		#region Event Handlers
		private void OnTriggerStay(Collider other)
		{
			if (_colliderCount == 0)
			{
				_colliderCount++;
			}
		}

		private void OnTriggerEnter(Collider other)
		{
			_colliderCount++;
		}

		private void OnTriggerExit(Collider other)
		{
			_colliderCount--;
		}

		private void OnPointerDownDelegate(PointerEventData data)
		{
			if (Array.IndexOf(keyInput, KeyCode.Mouse0) == -1) return;
			Pressed();
		}

		private void OnPointerUpDelegate(PointerEventData data)
		{
			if (Array.IndexOf(keyInput, KeyCode.Mouse0) == -1) return;
			Released();
		}

		private void OnPointerEnterDelegate(PointerEventData data)
		{
			Selected();
		}

		private void OnPointerExitDelegate(PointerEventData data)
		{
			Deselected();
		}

		private void OnMouseDown()
		{
			if (_vrRunning) return;
			if (Array.IndexOf(keyInput, KeyCode.Mouse0) == -1) return;
			Pressed();
		}

		private void OnMouseUp()
		{
			if (_vrRunning) return;
			if (Array.IndexOf(keyInput, KeyCode.Mouse0) == -1) return;
			Released();
			if (Application.isMobilePlatform)
			{
				Deselected();
			}
		}

		private void OnMouseEnter()
		{
			if (Application.isMobilePlatform) return;
			if (_vrRunning) return;
			Selected();
		}

		private void OnMouseExit()
		{
			if (_vrRunning) return;
			Deselected();
		}

		public virtual void Deselected()
		{			
			_selectedCount--;
			if (_selectedCount < 0) _selectedCount = 0;
			if (_selectedCount > 0) return;
			_clicking = false;
			ColorNormal();
			ScaleNormal();
			if (!Application.isMobilePlatform)
			{
				if (OnDeselected != null) OnDeselected.Invoke(gameObject);
				IsSelected = false;
			}
		}

		public virtual void Selected()
		{
			_selectedCount++;
			if (_selectedCount != 1) return;

			_pressed = false;
			_released = false;

			_clicking = false;
			ColorSelected();
			ScaleSelected();
			if (OnSelected != null) OnSelected.Invoke(gameObject);
			IsSelected = true;
		}

		public virtual void Pressed()
		{
			if (_selectedCount <= 0) return;
			if (_pressed) return;
			_pressed = true;
			_released = false;
			
			_clicking = true;
			ColorPressed();
			ScalePressed();
			if (OnPressed != null) OnPressed.Invoke(gameObject);
		}

		public virtual void Released()
		{
			if (_released) return;
			_pressed = false;
			_released = true;

			if (_selectedCount != 0)
			{
				ColorSelected();
				ScaleSelected();
			}

			if (_clicking)
			{
				if (OnClick != null) OnClick.Invoke(gameObject);
			}
			_clicking = false;
			if (OnReleased != null) OnReleased.Invoke(gameObject);
		}
		#endregion

		#region Private Methods
		private void ResizeGUIBoxCollider(BoxCollider boxCollider)
		{
			boxCollider.size = new Vector3(_rectTransform.rect.width + guiBoxColliderPadding.x, _rectTransform.rect.height + guiBoxColliderPadding.y, _boxCollider.size.z);

			float centerX = (Mathf.Abs(_rectTransform.pivot.x - 1) - .5f) * boxCollider.size.x;
			float centerY = (Mathf.Abs(_rectTransform.pivot.y - 1) - .5f) * boxCollider.size.y;
			boxCollider.center = new Vector3(centerX, centerY, boxCollider.center.z);
		}

		private void ColorReset()
		{
			if (_colorTween != null) _colorTween.Stop();
			if (colorRendererTarget != null) colorRendererTarget.material.color = normalColor;
			if (colorImageTarget != null) colorImageTarget.color = normalColor;
		}

		private void ColorNormal()
		{
			if (!applyColor) return;
			if (colorRendererTarget != null) _colorTween = Tween.Color(colorRendererTarget, normalColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
			if (colorImageTarget != null) _colorTween = Tween.Color(colorImageTarget, normalColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
		}

		private void ColorSelected()
		{
			if (!applyColor) return;
			if (colorRendererTarget != null) _colorTween = Tween.Color(colorRendererTarget, selectedColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
			if (colorImageTarget != null) _colorTween = Tween.Color(colorImageTarget, selectedColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
		}

		private void ColorPressed()
		{
			if (!applyColor) return;
			if (colorRendererTarget != null) _colorTween = Tween.Color(colorRendererTarget, pressedColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
			if (colorImageTarget != null) _colorTween = Tween.Color(colorImageTarget, pressedColor, colorDuration, 0, null, Tween.LoopType.None, null, null, false);
		}

		private void ScaleReset()
		{
			if (_scaleTween != null) _scaleTween.Stop();
			scaleTarget.localScale = normalScale;
		}

		private void ScaleNormal()
		{
			if (!applyScale) return;
			AnimationCurve curve = null;
			switch (scaleEaseType)
			{
				case EaseType.EaseOut:
					curve = Tween.EaseOutStrong;
					break;

				case EaseType.EaseOutBack:
					curve = Tween.EaseOutBack;
					break;
			}
			_scaleTween = Tween.LocalScale(scaleTarget, normalScale, scaleDuration, 0, curve, Tween.LoopType.None, null, null, false);
		}

		private void ScaleSelected()
		{
			if (!applyScale) return;
			AnimationCurve curve = null;
			switch (scaleEaseType)
			{
				case EaseType.EaseOut:
					curve = Tween.EaseOutStrong;
					break;

				case EaseType.EaseOutBack:
					curve = Tween.EaseOutBack;
					break;
			}
			_scaleTween = Tween.LocalScale(scaleTarget, selectedScale, scaleDuration, 0, curve, Tween.LoopType.None, null, null, false);
		}

		private void ScalePressed()
		{
			if (!applyScale) return;
			AnimationCurve curve = null;
			switch (scaleEaseType)
			{
				case EaseType.EaseOut:
					curve = Tween.EaseOutStrong;
					break;

				case EaseType.EaseOutBack:
					curve = Tween.EaseOutBack;
					break;
			}
			_scaleTween = Tween.LocalScale(scaleTarget, pressedScale, scaleDuration, 0, curve, Tween.LoopType.None, null, null, false);
		}
		#endregion
	}
}