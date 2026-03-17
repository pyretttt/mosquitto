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

private enum Section { case main }
private typealias Item = IdentifiableValue<UUID, OptionModel>

// MARK: - ViewController

@MainActor
final class OptionsViewController: UIViewController {

    private let onDismiss: @Sendable @MainActor () -> Void
    private var items: [Item] {
        didSet {
            itemsMap = Dictionary(uniqueKeysWithValues: items.map { ($0.left, $0.right) })
        }
    }
    private var itemsMap: [UUID: OptionModel]

    private let tableView = modify(UITableView(frame: .zero, style: .insetGrouped)) {
        $0.register(TextFieldOptionCell.self, forCellReuseIdentifier: TextFieldOptionCell.reuseId)
        $0.register(SwitchOptionCell.self, forCellReuseIdentifier: SwitchOptionCell.reuseId)
        $0.register(SegmentedOptionCell.self, forCellReuseIdentifier: SegmentedOptionCell.reuseId)
        $0.rowHeight = UITableView.automaticDimension
        $0.estimatedRowHeight = 56
        $0.backgroundColor = .systemGroupedBackground
    }
    private let closeButton = modify(UIButton(type: .system)) {
        $0.setImage(UIImage(systemName: "xmark"), for: .normal)
        $0.tintColor = .label
        $0.backgroundColor = .secondarySystemBackground
        $0.layer.cornerRadius = 16
        $0.accessibilityLabel = "Close options"
    }

    private lazy var dataSource = UITableViewDiffableDataSource<Section, UUID>(
        tableView: tableView
    ) { [weak self] tableView, indexPath, item in
        guard let self,
              let value = self.itemsMap[item] else { assertionFailure(); return UITableViewCell() }
        return value.cell(in: tableView, at: indexPath)
    }

    init(
        options: [OptionModel],
        onDismiss: @escaping @Sendable @MainActor () -> Void
    ) {
        self.onDismiss = onDismiss
        self.items = options.map { makeIdentifiable(value: $0) }
        self.itemsMap = Dictionary(uniqueKeysWithValues: items.map { ($0.left, $0.right) })
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemGroupedBackground
        view.addSubview(tableView)
        view.addSubview(closeButton)
        tableView.snp.makeConstraints { $0.edges.equalToSuperview() }
        closeButton.snp.makeConstraints {
            $0.top.equalTo(view.safeAreaLayoutGuide.snp.top).offset(8)
            $0.trailing.equalToSuperview().inset(16)
            $0.width.height.equalTo(32)
        }
        closeButton.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
        applySnapshot()
    }

    private func applySnapshot() {
        var snapshot = NSDiffableDataSourceSnapshot<Section, UUID>()
        snapshot.appendSections([.main])
        snapshot.appendItems(items.map(\.left))
        dataSource.apply(snapshot, animatingDifferences: false)
    }

    @objc
    private func closeTapped() {
        dismiss(animated: true, completion: onDismiss)
    }
}

// MARK: - Cell Provider

@MainActor
extension OptionModel {
    fileprivate func cell(in tableView: UITableView, at indexPath: IndexPath) -> UITableViewCell {
        switch self {
        case .int(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: TextFieldOptionCell.reuseId, for: indexPath
            ) as! TextFieldOptionCell
            cell.configure(name: name, text: String(ptr.pointee.value), keyboardType: .numberPad) { text in
                if let v = Int32(text) { 
                    ptr.pointee.value = v 
                }
            }
            return cell

        case .float(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: TextFieldOptionCell.reuseId, for: indexPath
            ) as! TextFieldOptionCell
            cell.configure(name: name, text: String(ptr.pointee.value), keyboardType: .decimalPad) { text in
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
                items: ptr.pointee.values.map { String($0) },
                selected: Int(ptr.pointee.selected)
            ) { idx in ptr.pointee.selected = size_t(idx) }
            return cell

        case .multiFloat(let name, let ptr):
            let cell = tableView.dequeueReusableCell(
                withIdentifier: SegmentedOptionCell.reuseId, for: indexPath
            ) as! SegmentedOptionCell
            cell.configure(
                name: name,
                items: ptr.pointee.values.map { String($0) },
                selected: Int(ptr.pointee.selected)
            ) { idx in ptr.pointee.selected = size_t(idx) }
            return cell
        }
    }
}

// MARK: - Cells

@MainActor
private final class TextFieldOptionCell: UITableViewCell {
    fileprivate static let reuseId = "TextFieldOptionCell"

    fileprivate var onChange: @MainActor @Sendable (String) -> Void = { _ in }

    private var nameLabel = modify(UILabel()) {
        $0.font = .systemFont(ofSize: 15, weight: .medium)
        $0.textColor = .secondaryLabel
    }
    private var textField = modify(UITextField()) {
        $0.borderStyle = .roundedRect
        $0.font = .systemFont(ofSize: 14)
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
            $0.leading.trailing.equalToSuperview().inset(24)
            $0.top.equalTo(nameLabel.snp.bottom).offset(4)
            $0.bottom.equalToSuperview().inset(10)
            $0.height.equalTo(44)
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
        onChange(text)
    }
}

@MainActor
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
        onChange(value)
    }
}

@MainActor
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

    @objc 
    private func segmentChanged() {
        let idx = segmented.selectedSegmentIndex
        onChange(idx)
    }
}
