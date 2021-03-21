package com.soloupis.sample.photos_with_depth.di

import com.soloupis.sample.photos_with_depth.fragments.segmentation.OcrViewModel
import com.soloupis.sample.photos_with_depth.fragments.segmentation.OcrModelExecutor
import org.koin.android.viewmodel.dsl.viewModel
import org.koin.dsl.module

val segmentationAndStyleTransferModule = module {

    factory { OcrModelExecutor(get(), false) }

    viewModel {
        OcrViewModel(get())
    }
}