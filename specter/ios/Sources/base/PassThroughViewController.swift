//
//  PassThroughViewController.swift
//  app
//
//  Created by Codex on 04.03.2026.
//

import UIKit

/// A view controller whose root view ignores touches so they pass through to views behind it.
open class PassThroughViewController: UIViewController {
    open override func loadView() {
        view = modify(PassThroughView()) {
            $0.backgroundColor = .clear
        }
    }
}

open class PassThroughView: UIView {
    open override func hitTest(
        _ point: CGPoint,
        with event: UIEvent?
    ) -> UIView? {
        let hitView = super.hitTest(point, with: event)
        return hitView === self ? nil : hitView
    }
}
