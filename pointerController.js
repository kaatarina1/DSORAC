export class PointerController extends EventTarget {
	constructor(emitter = document.body) {
		super();

		this.emitter = emitter;

		this.pointerdownHandler = this.pointerdownHandler.bind(this);
		this.pointerupHandler = this.pointerupHandler.bind(this);
		this.pointermoveHandler = this.pointermoveHandler.bind(this);
		this.wheelHandler = this.wheelHandler.bind(this);

		this.emitter.addEventListener("pointerdown", this.pointerdownHandler);
		this.emitter.addEventListener("wheel", this.wheelHandler, {
			passive: false,
		}); // Make sure to prevent default scroll behavior
	}

	pointerdownHandler(e) {
		this.emitter.setPointerCapture(e.pointerId);
		this.emitter.requestPointerLock();
		this.emitter.removeEventListener(
			"pointerdown",
			this.pointerdownHandler
		);
		this.emitter.addEventListener("pointerup", this.pointerupHandler);
		this.emitter.addEventListener("pointermove", this.pointermoveHandler);
		this.dispatchEvent(new PointerEvent("pointerdown", e));
	}

	pointerupHandler(e) {
		this.emitter.releasePointerCapture(e.pointerId);
		this.emitter.ownerDocument.exitPointerLock();
		this.emitter.addEventListener("pointerdown", this.pointerdownHandler);
		this.emitter.removeEventListener("pointerup", this.pointerupHandler);
		this.emitter.removeEventListener(
			"pointermove",
			this.pointermoveHandler
		);
		this.dispatchEvent(new PointerEvent("pointerup", e));
	}

	pointermoveHandler(e) {
		this.dispatchEvent(new PointerEvent("pointermove", e));
	}

	// Handle mouse wheel scroll (zoom)
	wheelHandler(e) {
		e.preventDefault(); // Prevent default scrolling behavior to allow zoom
		this.dispatchEvent(new WheelEvent("wheel", e));
	}
}
