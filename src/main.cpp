#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct RawSample {
    int sample_id{};
    double speed{};
    double accel{};
    double yaw_rate{};
    int label{};
};

struct FeatureRecord {
    int sample_id{};
    std::vector<double> features;
    int label{};
};

struct PredictionRecord {
    int sample_id{};
    double score{};
    double probability{};
    int prediction{};
    int label{};
};

std::vector<std::string> Split(const std::string& line, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string item;

    while (std::getline(ss, item, delimiter)) {
        result.push_back(item);
    }

    return result;
}

std::vector<RawSample> LoadSamplesFromCsv(const std::string& csv_path) {
    std::ifstream fin(csv_path);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open input csv: " + csv_path);
    }

    std::vector<RawSample> samples;
    std::string line;

    // 跳过表头
    std::getline(fin, line);

    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }

        auto cols = Split(line, ',');
        if (cols.size() != 5) {
            std::cerr << "Skip invalid line: " << line << std::endl;
            continue;
        }

        RawSample sample;
        sample.sample_id = std::stoi(cols[0]);
        sample.speed = std::stod(cols[1]);
        sample.accel = std::stod(cols[2]);
        sample.yaw_rate = std::stod(cols[3]);
        sample.label = std::stoi(cols[4]);

        samples.push_back(sample);
    }

    return samples;
}

FeatureRecord ExtractFeatures(const RawSample& sample) {
    FeatureRecord record;
    record.sample_id = sample.sample_id;
    record.label = sample.label;

    // 模拟特征工程
    // 注意：真实项目中这里可能是图像特征、轨迹特征、点云特征、规划特征等
    double speed_norm = sample.speed / 60.0;
    double accel_norm = sample.accel / 5.0;
    double yaw_rate_norm = sample.yaw_rate / 0.3;
    double speed_square = speed_norm * speed_norm;
    double risk_motion = std::abs(accel_norm) + std::abs(yaw_rate_norm);
    double speed_accel_interaction = speed_norm * accel_norm;

    record.features = {
        speed_norm,
        accel_norm,
        yaw_rate_norm,
        speed_square,
        risk_motion,
        speed_accel_interaction
    };

    return record;
}

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

PredictionRecord RunFakeModel(const FeatureRecord& feature_record) {
    const auto& f = feature_record.features;

    // 模拟一个线性模型：
    // score = w0*x0 + w1*x1 + ...
    // 真实项目中这里会替换为 ONNX Runtime / TensorRT / 自研 C++ 模型推理模块
    double score =
        1.2 * f[0] +
        1.8 * f[1] +
        1.5 * f[2] +
        0.8 * f[3] +
        2.0 * f[4] -
        2.0;

    double probability = Sigmoid(score);
    int prediction = probability >= 0.5 ? 1 : 0;

    PredictionRecord pred;
    pred.sample_id = feature_record.sample_id;
    pred.score = score;
    pred.probability = probability;
    pred.prediction = prediction;
    pred.label = feature_record.label;

    return pred;
}

void ExportFeaturesToCsv(
    const std::vector<FeatureRecord>& records,
    const std::string& output_path
) {
    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open output csv: " + output_path);
    }

    fout << "sample_id,speed_norm,accel_norm,yaw_rate_norm,speed_square,risk_motion,label\n";

    for (const auto& record : records) {
        fout << record.sample_id;

        for (double value : record.features) {
            fout << "," << std::fixed << std::setprecision(6) << value;
        }

        fout << "," << record.label << "\n";
    }
}

void ExportPredictionsToJsonl(
    const std::vector<PredictionRecord>& records,
    const std::string& output_path
) {
    std::ofstream fout(output_path);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open output jsonl: " + output_path);
    }

    for (const auto& record : records) {
        fout << "{"
             << "\"sample_id\":" << record.sample_id << ","
             << "\"score\":" << std::fixed << std::setprecision(6) << record.score << ","
             << "\"probability\":" << std::fixed << std::setprecision(6) << record.probability << ","
             << "\"prediction\":" << record.prediction << ","
             << "\"label\":" << record.label
             << "}\n";
    }
}

void ExportFeaturesToBinary(
    const std::vector<FeatureRecord>& records,
    const std::string& output_path
) {
    std::ofstream fout(output_path, std::ios::binary);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open output binary: " + output_path);
    }

    int magic_number = 20260427;
    int version = 1;
    int sample_count = static_cast<int>(records.size());
    int feature_dim = records.empty() ? 0 : static_cast<int>(records[0].features.size());

    fout.write(reinterpret_cast<const char*>(&magic_number), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&version), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&sample_count), sizeof(int));
    fout.write(reinterpret_cast<const char*>(&feature_dim), sizeof(int));

    for (const auto& record : records) {
        fout.write(reinterpret_cast<const char*>(&record.sample_id), sizeof(int));

        for (double value : record.features) {
            float v = static_cast<float>(value);
            fout.write(reinterpret_cast<const char*>(&v), sizeof(float));
        }

        fout.write(reinterpret_cast<const char*>(&record.label), sizeof(int));
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cout << "Usage:\n"
                      << "  model_export <input_csv> <output_dir>\n\n"
                      << "Example:\n"
                      << "  model_export data/samples.csv output\n";
            return 1;
        }

        std::string input_csv = argv[1];
        std::string output_dir = argv[2];

        std::filesystem::create_directories(output_dir);

        std::cout << "[1] Loading samples from: " << input_csv << std::endl;
        auto samples = LoadSamplesFromCsv(input_csv);

        std::cout << "[2] Extracting features..." << std::endl;
        std::vector<FeatureRecord> feature_records;
        feature_records.reserve(samples.size());

        for (const auto& sample : samples) {
            feature_records.push_back(ExtractFeatures(sample));
        }

        std::cout << "[3] Running fake model..." << std::endl;
        std::vector<PredictionRecord> prediction_records;
        prediction_records.reserve(feature_records.size());

        for (const auto& feature_record : feature_records) {
            prediction_records.push_back(RunFakeModel(feature_record));
        }

        std::cout << "[4] Exporting files..." << std::endl;

        ExportFeaturesToCsv(
            feature_records,
            output_dir + "/features.csv"
        );

        ExportPredictionsToJsonl(
            prediction_records,
            output_dir + "/predictions.jsonl"
        );

        ExportFeaturesToBinary(
            feature_records,
            output_dir + "/features.bin"
        );

        std::cout << "\nExport finished successfully.\n";
        std::cout << "Generated files:\n";
        std::cout << "  " << output_dir << "/features.csv\n";
        std::cout << "  " << output_dir << "/predictions.jsonl\n";
        std::cout << "  " << output_dir << "/features.bin\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
