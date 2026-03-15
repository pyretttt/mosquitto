//
//  OptionsViewController.swift
//  app
//
//  Created by Бакулин Семен Александрович on 15.03.2026.
//

import UIKit
import SnapKit
import core_cpp
import ios_Base
import CxxStdlib

// MARK: - Model

enum OptionModel {
    case int(name: String, ptr: cv.IntOptionPtr)
    case float(name: String, ptr: cv.FloatOptionPtr)
    case bool(name: String, ptr: cv.BoolOptionPtr)
    case string(name: String, ptr: cv.StringOptionPtr)
    case multiString(name: String, ptr: cv.MultiStringOptionPtr)
    case multiInt(name: String, ptr: cv.MultiIntegerOptionPtr)
    case multiFloat(name: String, ptr: cv.MultiFloatOptionPtr)
}

// TODO(human): implement buildModels(from:) — iterate the C++ vector, read each
// element's `.kind()` to identify its concrete type, cast via `cv.asInt` / `cv.asFloat`
// etc., and return the corresponding OptionModel case.
// Use the option's `.name` field (a std::string bridged to Swift String) as the label.
//
// Signature:
// private func buildModels(
//     from options: std.vector<std.__shared_ptr<cv.BaseOption>>
// ) -> [OptionModel]

// MARK: - ViewController

@MainActor
final class OptionsViewController: UIViewController {

    private var options: [OptionModel]

    private lazy var tableView = modify(UITableView(frame: .zero, style: .insetGrouped)) {
        $0.register(TextFieldOptionCell.self, forCellReuseIdentifier: TextFieldOptionCell.reuseId)
        $0.register(SwitchOptionCell.self, forCellReuseIdentifier: SwitchOptionCell.reuseId)
        $0.register(SegmentedOptionCell.self, forCellReuseIdentifier: SegmentedOptionCell.reuseId)
        $0.dataSource = self
        $0.rowHeight = UITableView.automaticDimension
        $0.estimatedRowHeight = 56
        $0.backgroundColor = .systemGroupedBackground
    }

    init(options: [OptionModel]) {
        self.options = options
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemGroupedBackground
        view.addSubview(tableView)
        tableView.snp.makeConstraints { $0.edges.equalToSuperview() }
    }
}

// MARK: - UITableViewDataSource

extension OptionsViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        options.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        switch options[indexPath.row] {
        case .int(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: TextFieldOptionCell.reuseId, for: indexPath
            ) as! TextFieldOptionCell
            cell.configure(name: name, text: "\(ptr.pointee.value)", keyboardType: .numberPad) { text in
                if let v = Int32(text) { ptr.pointee.value = v }
            }
            return cell

        case .float(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: TextFieldOptionCell.reuseId, for: indexPath
            ) as! TextFieldOptionCell
            cell.configure(name: name, text: "\(ptr.pointee.value)", keyboardType: .decimalPad) { text in
                if let v = Float(text) { ptr.pointee.value = v }
            }
            return cell

        case .bool(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: SwitchOptionCell.reuseId, for: indexPath
            ) as! SwitchOptionCell
            cell.configure(name: name, isOn: ptr.pointee.value) { isOn in
                ptr.pointee.value = isOn
            }
            return cell

        case .string(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: TextFieldOptionCell.reuseId, for: indexPath
            ) as! TextFieldOptionCell
            cell.configure(name: name, text: String(ptr.pointee.value), keyboardType: .default) { text in
                ptr.pointee.value = std.string(text)
            }
            return cell

        case .multiString(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: SegmentedOptionCell.reuseId, for: indexPath
            ) as! SegmentedOptionCell
            cell.configure(
                name: name,
                items: ptr.pointee.values.map { String($0) },
                selected: Int(ptr.pointee.selected)
            ) { idx in ptr.pointee.selected = size_t(idx) }
            return cell

        case .multiInt(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: SegmentedOptionCell.reuseId, for: indexPath
            ) as! SegmentedOptionCell
            cell.configure(
                name: name,
                items: ptr.pointee.values.map { "\($0)" },
                selected: Int(ptr.pointee.selected)
            ) { idx in ptr.pointee.selected = size_t(idx) }
            return cell

        case .multiFloat(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: SegmentedOptionCell.reuseId, for: indexPath
            ) as! SegmentedOptionCell
            cell.configure(
                name: name,
                items: ptr.pointee.values.map { "\($0)" },
                selected: Int(ptr.pointee.selected)
            ) { idx in ptr.pointee.selected = size_t(idx) }
            return cell
        }
    }
}

// MARK: - Cells

private final class TextFieldOptionCell: UITableViewCell {
    fileprivate static let reuseId = "TextFieldOptionCell"

    fileprivate var onChange: @MainActor @Sendable (String) -> Void = { _ in }

    private var nameLabel = modify(UILabel()) {
        $0.font = .systemFont(ofSize: 13, weight: .medium)
        $0.textColor = .secondaryLabel
    }
    private var textField = modify(UITextField()) {
        $0.borderStyle = .roundedRect
        $0.font = .systemFont(ofSize: 15)
        $0.clearButtonMode = .whileEditing
    }

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        selectionStyle = .none
        contentView.addSubview(nameLabel)
        contentView.addSubview(textField)
        nameLabel.snp.makeConstraints {
            $0.leading.trailing.equalToSuperview().inset(16)
            $0.top.equalToSuperview().inset(10)
        }
        textField.snp.makeConstraints {
            $0.leading.trailing.equalToSuperview().inset(16)
            $0.top.equalTo(nameLabel.snp.bottom).offset(4)
            $0.bottom.equalToSuperview().inset(10)
            $0.height.equalTo(36)
        }
        textField.addTarget(self, action: #selector(textChanged), for: .editingChanged)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    fileprivate func configure(
        name: String,
        text: String,
        keyboardType: UIKeyboardType,
        onChange: @escaping @MainActor @Sendable (String) -> Void
    ) {
        nameLabel.text = name
        textField.text = text
        textField.keyboardType = keyboardType
        self.onChange = onChange
    }

    @objc private func textChanged() {
        let text = textField.text ?? ""
        Task { @MainActor in onChange(text) }
    }
}

private final class SwitchOptionCell: UITableViewCell {
    fileprivate static let reuseId = "SwitchOptionCell"

    fileprivate var onChange: @MainActor @Sendable (Bool) -> Void = { _ in }

    private var nameLabel = modify(UILabel()) {
        $0.font = .systemFont(ofSize: 15)
    }
    private var toggle = UISwitch()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        selectionStyle = .none
        contentView.addSubview(nameLabel)
        contentView.addSubview(toggle)
        nameLabel.snp.makeConstraints {
            $0.leading.equalToSuperview().inset(16)
            $0.centerY.equalToSuperview()
            $0.trailing.lessThanOrEqualTo(toggle.snp.leading).offset(-8)
        }
        toggle.snp.makeConstraints {
            $0.trailing.equalToSuperview().inset(16)
            $0.centerY.equalToSuperview()
            $0.top.bottom.equalToSuperview().inset(10)
        }
        toggle.addTarget(self, action: #selector(toggled), for: .valueChanged)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    fileprivate func configure(
        name: String,
        isOn: Bool,
        onChange: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        nameLabel.text = name
        toggle.isOn = isOn
        self.onChange = onChange
    }

    @objc private func toggled() {
        let value = toggle.isOn
        Task { @MainActor in onChange(value) }
    }
}

private final class SegmentedOptionCell: UITableViewCell {
    fileprivate static let reuseId = "SegmentedOptionCell"

    fileprivate var onChange: @MainActor @Sendable (Int) -> Void = { _ in }

    private var nameLabel = modify(UILabel()) {
        $0.font = .systemFont(ofSize: 13, weight: .medium)
        $0.textColor = .secondaryLabel
    }
    private var segmented = UISegmentedControl()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        selectionStyle = .none
        contentView.addSubview(nameLabel)
        contentView.addSubview(segmented)
        nameLabel.snp.makeConstraints {
            $0.leading.trailing.equalToSuperview().inset(16)
            $0.top.equalToSuperview().inset(10)
        }
        segmented.snp.makeConstraints {
            $0.leading.trailing.equalToSuperview().inset(16)
            $0.top.equalTo(nameLabel.snp.bottom).offset(6)
            $0.bottom.equalToSuperview().inset(10)
        }
        segmented.addTarget(self, action: #selector(segmentChanged), for: .valueChanged)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    fileprivate func configure(
        name: String,
        items: [String],
        selected: Int,
        onChange: @escaping @MainActor @Sendable (Int) -> Void
    ) {
        nameLabel.text = name
        segmented.removeAllSegments()
        items.enumerated().forEach { segmented.insertSegment(withTitle: $1, at: $0, animated: false) }
        segmented.selectedSegmentIndex = selected
        self.onChange = onChange
    }

    @objc private func segmentChanged() {
        let idx = segmented.selectedSegmentIndex
        Task { @MainActor in onChange(idx) }
    }
}

private func buildModels(
    from options: cv.OptionList
) -> [OptionModel] {
    options.map {
        let name = String($0.pointee.name)
        return switch $0.pointee.kind() {
        case cv.OptionKind.Int:
            OptionModel.int(name: name, ptr: cv.asInt($0))
        case .Float:
            OptionModel.float(name: name, ptr: cv.asFloat($0))
        case .Bool:
            OptionModel.bool(name: name, ptr: cv.asBool($0))
        case .String:
            OptionModel.string(name: name, ptr: cv.asString($0))
        case .MultiString:
            OptionModel.multiString(name: name, ptr: cv.asMultiString($0))
        case .MultiInteger:
            OptionModel.multiInt(name: name, ptr: cv.asMultiInteger($0))
        case .MultiFloat:
            OptionModel.multiFloat(name: name, ptr: cv.asMultiFloat($0))
        @unknown default:
            fatalError()
        }
    }
}
