class CarForm {
	constructor() {
		this.form = document.getElementById("carForm");
		this.brandSelect = document.getElementById("brand");
		this.modelSelect = document.getElementById("model");
		this.resultContainer = document.getElementById("resultContainer");
		this.resultPrice = document.getElementById("resultPrice");
		this.submitBtn = document.getElementById("submitBtn");

		this.init();
	}

	init() {
		this.brandSelect.addEventListener("change", () => this.onBrandChange());

		this.form.addEventListener("submit", (e) => this.onSubmit(e));
	}

	async onBrandChange() {
		const brand = this.brandSelect.value;

		if (!brand) {
			this.resetModelSelect();
			return;
		}

		try {
			const response = await fetch(
				`/api/models/${encodeURIComponent(brand.toLowerCase())}`,
			);

			if (!response.ok) {
				throw new Error(`Failed to fetch models: ${response.statusText}`);
			}

			const data = await response.json();

			this.updateModelSelect(data.models);
		} catch (error) {
			console.error("Error fetching models:", error);
			this.showError("Error loading models. Please try again.");
			this.resetModelSelect();
		}
	}

	resetModelSelect() {
		this.modelSelect.innerHTML = '<option value="">Select Model</option>';
		this.modelSelect.disabled = true;
	}

	updateModelSelect(models) {
		this.modelSelect.innerHTML = '<option value="">Select Model</option>';

		models.forEach((model) => {
			const option = document.createElement("option");
			option.value = model;
			option.textContent = this.capitalizeModel(model);
			this.modelSelect.appendChild(option);
		});

		this.modelSelect.disabled = false;
	}

	capitalizeModel(model) {
		return model
			.split(" ")
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join(" ")
			.split("-")
			.map((word) => word.charAt(0).toUpperCase() + word.slice(1))
			.join("-");
	}

	onSubmit(event) {
		event.preventDefault();

		const formData = new FormData(this.form);
		const data = {};

		for (const [key, value] of formData.entries()) {
			data[key]  = value
		}

		if (!data.has_damage) {
			data.has_damage = false;
		}

		this.submitForm(data);
	}

	async submitForm(data) {
		this.setLoadingState(true);

		try {
			const response = await fetch("/submit", {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(data),
			});

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			const result = await response.json();

			this.displayResult(result);

			this.clearForm();
		} catch (error) {
			console.error("Error:", error);
			this.showError("Error submitting form. Please try again.");
		} finally {
			this.setLoadingState(false);
		}
	}

	setLoadingState(isLoading) {
		if (isLoading) {
			this.submitBtn.disabled = true;
			this.submitBtn.classList.add("loading");
		} else {
			this.submitBtn.disabled = false;
			this.submitBtn.classList.remove("loading");
		}
	}

	showError(message) {
		this.resultPrice.textContent = message;
		this.resultPrice.style.color = "#ef4444";
		this.resultContainer.style.display = "block";

		this.resultContainer.scrollIntoView({
			behavior: "smooth",
			block: "nearest",
		});

		setTimeout(() => {
			this.resultPrice.style.color = "#ffffff";
		}, 3000);
	}

	displayResult(result) {
		const price =
			result.price || "N/A";

		const formattedPrice =
			typeof price === "number"
				? `£${price.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
				: price;

		this.resultPrice.textContent = formattedPrice;
		this.resultPrice.style.color = "#ffffff";
		this.resultContainer.style.display = "block";

		this.resultContainer.scrollIntoView({
			behavior: "smooth",
			block: "nearest",
		});
	}

	clearForm() {
		this.form.reset();

		this.resetModelSelect();
	}
}

document.addEventListener("DOMContentLoaded", () => {
	new CarForm();
});
