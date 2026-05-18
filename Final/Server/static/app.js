const form = document.querySelector("#analysisForm");
const feedbackText = document.querySelector("#feedbackText");
const threshold = document.querySelector("#threshold");
const thresholdValue = document.querySelector("#thresholdValue");
const submitButton = document.querySelector("#submitButton");
const sampleButton = document.querySelector("#sampleButton");
const apiStatus = document.querySelector("#apiStatus");
const resultSummary = document.querySelector("#resultSummary");
const resultMeta = document.querySelector("#resultMeta");
const aspectCount = document.querySelector("#aspectCount");
const metaThreshold = document.querySelector("#metaThreshold");
const dominantSentiment = document.querySelector("#dominantSentiment");
const emptyState = document.querySelector("#emptyState");
const resultsList = document.querySelector("#resultsList");

const sampleText = "Giảng viên dạy rất hay nhưng phòng học quá nóng.";
const sentimentLabels = {
  positive: "Tích cực",
  negative: "Tiêu cực",
  neutral: "Trung lập",
};

const categoryLabels = {
  ky_nang_giang_day: "Kỹ năng giảng dạy",
  hanh_vi: "Hành vi",
  de_xuat: "Đề xuất",
  bai_tap: "Bài tập",
  chuong_trinh_hoc: "Chương trình học",
  kien_thuc: "Kiến thức",
  kinh_nghiem: "Kinh nghiệm",
  cung_cap_tai_lieu: "Cung cấp tài liệu",
  thiet_bi_day_hoc: "Thiết bị dạy học",
  cham_diem: "Chấm điểm",
  noi_chung: "Nói chung",
};

function formatThreshold(value) {
  return Number(value).toFixed(2);
}

function setApiStatus(state, text) {
  apiStatus.classList.toggle("is-ok", state === "ok");
  apiStatus.classList.toggle("is-error", state === "error");
  apiStatus.querySelector("span:last-child").textContent = text;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function getDominantSentiment(results) {
  const counts = results.reduce((acc, item) => {
    acc[item.sentiment] = (acc[item.sentiment] || 0) + 1;
    return acc;
  }, {});

  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "-";
}

function renderResults(data) {
  const results = Array.isArray(data.results) ? data.results : [];

  resultsList.innerHTML = "";
  emptyState.hidden = results.length > 0;
  resultMeta.hidden = false;
  aspectCount.textContent = results.length;
  metaThreshold.textContent = formatThreshold(data.threshold);

  const dominant = getDominantSentiment(results);
  dominantSentiment.textContent = sentimentLabels[dominant] || "-";

  if (results.length === 0) {
    resultSummary.textContent = "Không có category nào vượt threshold hiện tại.";
    emptyState.hidden = false;
    emptyState.querySelector("p").textContent = "Thử giảm threshold hoặc nhập phản hồi rõ hơn để xem thêm kết quả.";
    return;
  }

  resultSummary.textContent = `Mô hình tìm thấy ${results.length} khía cạnh trong phản hồi này.`;

  results.forEach((item) => {
    const sentiment = item.sentiment || "neutral";
    const confidence = Math.max(0, Math.min(1, Number(item.confidence) || 0));
    const percent = Math.round(confidence * 100);
    const category = categoryLabels[item.category] || item.category || "Không rõ";

    const card = document.createElement("article");
    card.className = `result-card ${sentiment}`;
    card.innerHTML = `
      <div class="result-card-header">
        <p class="category-name">${escapeHtml(category)}</p>
        <span class="sentiment-badge ${escapeHtml(sentiment)}">${escapeHtml(sentimentLabels[sentiment] || sentiment)}</span>
      </div>
      <!-- COMMENTED: Hiển thị thanh % confidence của sentiment
      <div class="confidence-row">
        <div class="confidence-track" aria-hidden="true">
          <div class="confidence-fill" style="width: ${percent}%"></div>
        </div>
        <span>${percent}%</span>
      </div>
      -->
    `;
    resultsList.appendChild(card);
  });
}

function renderError(message) {
  resultMeta.hidden = true;
  emptyState.hidden = true;
  resultsList.innerHTML = `<div class="error-box">${escapeHtml(message)}</div>`;
  resultSummary.textContent = "Không thể phân tích phản hồi.";
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("API chưa sẵn sàng");
    }
    setApiStatus("ok", "API sẵn sàng");
  } catch (error) {
    setApiStatus("error", "Không kết nối được API");
  }
}

threshold.addEventListener("input", () => {
  thresholdValue.textContent = formatThreshold(threshold.value);
});

sampleButton.addEventListener("click", () => {
  feedbackText.value = sampleText;
  feedbackText.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = feedbackText.value.trim();
  if (!text) {
    renderError("Vui lòng nhập nội dung cần phân tích.");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "Đang phân tích...";
  resultSummary.textContent = "Mô hình đang xử lý phản hồi.";
  emptyState.hidden = false;
  emptyState.querySelector("p").textContent = "Đang gửi dữ liệu tới /predict...";
  resultsList.innerHTML = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        threshold: Number(threshold.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request phân tích thất bại.");
    }

    renderResults(data);
  } catch (error) {
    renderError(error.message || "Có lỗi xảy ra khi gọi API.");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Phân tích";
  }
});

thresholdValue.textContent = formatThreshold(threshold.value);
checkHealth();
